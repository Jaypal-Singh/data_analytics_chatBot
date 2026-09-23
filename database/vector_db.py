import os
import faiss
import pickle
import numpy as np
import requests
from core.config import VECTOR_DB_PATH, EMBEDDING_MODEL_NAME, OLLAMA_EMBED_URL




class VectorDBManager:
    def __init__(self, user_id: str, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self.user_id = user_id
        self.vectordb_path = f"{VECTOR_DB_PATH}/{user_id}"
        self.model_name = embedding_model_name
        self.dimension = self._get_dimension()

        # Index + metadata initialize karo
        self.index = None
        self.metadata_store = []
        self._load_or_create_index()

    # ─────────────── Index Load / Create ───────────────

    def _load_or_create_index(self):
        """Agar pehle se saved index hai toh load karo, warna naya banao."""
        index_file = f"{self.vectordb_path}/index.faiss"
        meta_file = f"{self.vectordb_path}/metadata.pkl"

        if os.path.exists(index_file) and os.path.exists(meta_file):
            self.index = faiss.read_index(index_file)
            with open(meta_file, "rb") as f:
                self.metadata_store = pickle.load(f)
        else:
            # Naya khali index — IndexFlatIP = Cosine Similarity (normalized vectors ke saath)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata_store = []

    # ─────────────── Embed (Ollama API) ───────────────

    def _get_dimension(self) -> int:
        """Pehli baar ek dummy embed karke dimension pata karo."""
        test_embed = self._embed_single("test")
        return len(test_embed)

    def _embed_single(self, text: str) -> list:
        """Ek text ka embedding Ollama se lo."""
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": self.model_name, "input": text}
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

    def _embed(self, texts: list) -> np.ndarray:
        """Texts ko vectors mein convert karo + normalize (cosine similarity ke liye)."""
        embeddings = []
        for text in texts:
            embeddings.append(self._embed_single(text))
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    # ─────────────── Add Documents ───────────────

    def add_documents(self, texts: list, metadatas: list = None, doc_id: str = "default") -> str:
        """
        Documents embed karke index mein daalo.

        Args:
            texts: ["OPEX means expenses", "REV is net revenue"]
            metadatas: [{"type": "column_meaning"}, {"type": "few_shot"}]  (optional)
            doc_id: kis file/upload se aaya — "sales.csv", "data_dict.txt", etc.

        Returns:
            Status message
        """
        if metadatas and len(texts) != len(metadatas):
            raise ValueError("texts aur metadatas ki length match honi chahiye")

        if metadatas is None:
            metadatas = [{} for _ in texts]

        embeddings = self._embed(texts)
        self.index.add(embeddings)

        for text, meta in zip(texts, metadatas):
            meta_copy = meta.copy()
            meta_copy["text"] = text
            meta_copy["user_id"] = self.user_id
            meta_copy["doc_id"] = doc_id
            self.metadata_store.append(meta_copy)

        self._save()
        return f"Added {len(texts)} documents. Total vectors: {self.index.ntotal}"

    # ─────────────── Search ───────────────

    def search(self, query: str, top_k: int = 5, doc_id: str = None, filter_type: str = None) -> list:
        """
        Question pucho, similar documents milenge.

        Args:
            query: "What is OPEX_V2?"
            top_k: kitne results chahiye
            doc_id: optional — sirf ek specific dataset mein search
            filter_type: optional — "column_meaning" ya "few_shot"

        Returns:
            [{"text": "...", "score": 0.95, "doc_id": "...", ...}, ...]
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = self._embed([query])

        # Extra candidates nikalo — filter ke baad kam reh sakte hain
        search_k = min(self.index.ntotal, max(top_k * 5, 20))
        scores, indices = self.index.search(query_embedding, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata_store[idx]

            # Filters
            if meta.get("user_id") != self.user_id:
                continue
            if doc_id and meta.get("doc_id") != doc_id:
                continue
            if filter_type and meta.get("type") != filter_type:
                continue

            results.append({**meta, "score": float(score)})
            if len(results) >= top_k:
                break

        return results

    # ─────────────── Delete ───────────────

    def delete_by_doc(self, doc_id: str) -> str:
        """
        Ek document ke sare vectors hatao (jaise re-upload pe purana data delete karo).
        FAISS IndexFlatIP mein direct delete nahi hota — rebuild karna padta hai.
        """
        keep_indices = [
            i for i, meta in enumerate(self.metadata_store)
            if not (meta.get("user_id") == self.user_id and meta.get("doc_id") == doc_id)
        ]

        removed = len(self.metadata_store) - len(keep_indices)
        if removed == 0:
            return "Nothing to delete."

        old_metadata = self.metadata_store

        # Naya index banao bache hue vectors ke saath
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = []

        if keep_indices:
            texts_to_reembed = [old_metadata[i]["text"] for i in keep_indices]
            embeddings = self._embed(texts_to_reembed)
            self.index.add(embeddings)
            self.metadata_store = [old_metadata[i] for i in keep_indices]

        self._save()
        return f"Deleted {removed} vectors. Remaining: {self.index.ntotal}"

    # ─────────────── Save / Load ───────────────

    def _save(self):
        """Index + metadata disk pe save karo."""
        os.makedirs(self.vectordb_path, exist_ok=True)
        faiss.write_index(self.index, f"{self.vectordb_path}/index.faiss")
        with open(f"{self.vectordb_path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata_store, f)