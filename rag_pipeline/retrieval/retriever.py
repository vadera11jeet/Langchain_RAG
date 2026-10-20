from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List, Tuple
from sqlalchemy import Engine, text, bindparam
from rag_pipeline.vectorstore.vector_store import build_hybrid_search_config


class Retriever:

    def __init__(self, vector_store: VectorStore):
        self.vector_store: VectorStore = vector_store

    async def similarity_search(
        self, query: str, k: int = 20
    ) -> List[Tuple[Document, float]]:
        return await self.vector_store.asimilarity_search_with_score(
            query, k=k, hybrid_search_config=build_hybrid_search_config(query)
        )

    async def parent_child_retrieval(
        self, query, engine: Engine
    ) -> List[Tuple[List[Tuple[str, str]], List[Tuple[Document, float]]]]:

        child_docs = await self.similarity_search(query=query, k=25)

        parent_ids = list({child.metadata["parent_id"] for child, _ in child_docs})

        parent_content = await self.parent_retrieval(
            engine=engine, parent_ids=parent_ids
        )

        return parent_content, child_docs

    async def parent_retrieval(
        self, engine: Engine, parent_ids: List[str]
    ) -> List[Tuple[str, str]]:
        query = text("""
                    SELECT id, content FROM parent_documents WHERE id IN :parent_ids
                """).bindparams(bindparam("parent_ids", expanding=True))

        with engine.begin() as connection:

            result = connection.execute(query, {"parent_ids": parent_ids})

            parent_content = result.all()

        return parent_content
