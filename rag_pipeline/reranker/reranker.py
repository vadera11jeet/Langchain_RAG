from langchain_core.documents import Document
from typing import List, Tuple
from sentence_transformers import CrossEncoder


class ReRanker:

    def __init__(self):
        self.cross_encoder_model: CrossEncoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def cross_encoding_re_ranker(
        self, document_list: List[Tuple[Document, float]], query: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:

        if not document_list:
            return []

        pairs = [(query, doc.page_content) for doc, _ in document_list]

        scores = self.cross_encoder_model.predict(pairs)

        ranked = sorted(
            [(doc, float(score)) for (doc, _), score in zip(document_list, scores)],
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked[:top_k]
