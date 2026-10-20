import os
import json
from typing import List
from langchain_core.documents import Document
from langchain_postgres import PGVectorStore, PGEngine
from langchain_openai import AzureOpenAIEmbeddings
from langchain_postgres.v2.indexes import HNSWIndex
from langchain_core.vectorstores import VectorStore
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)
from sqlalchemy import Engine, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import create_async_engine


def build_hybrid_search_config(
    fts_query: str = "", index_name: str = "langchain_tsv_index"
) -> HybridSearchConfig:
    # the library writes the first query into config.fts_query and never clears it,
    # so a shared config makes every later search reuse the first query's keywords.
    # build a fresh config per search and pass it as hybrid_search_config=...
    return HybridSearchConfig(
        # index names are unique per schema, so each table needs its own
        index_name=index_name,
        # empty = no tsvector column; search/index use to_tsvector(lang, content)
        tsv_column="",
        tsv_lang="pg_catalog.english",
        fts_query=fts_query,
        primary_top_k=25,
        secondary_top_k=25,
        fusion_function=reciprocal_rank_fusion,
        fusion_function_parameters={"rrf_k": 60, "fetch_top_k": 10},
    )


class VectorStore:

    def __init__(self, collection_name: str = "collection_1"):
        self.collection_name = collection_name
        self.embedding = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        )

    async def storing_parent_child_embeddings(
        self,
        engine: Engine,
        connection_string: str,
        parent_chunks: List[Document],
        children_chunks: List[Document],
        table_name: str = "parent_child_embeddings",
        index_name: str = "employee_handbook_hnsw",
    ) -> VectorStore:
        """
        Embedding has been indexed in the PGVector vector db
        """

        vector_store = await self.create_embeddings(
            connection_string=connection_string,
            chunks=children_chunks,
            table_name=table_name,
            index_name=index_name,
        )

        self.storing_parents(engine=engine, docs=parent_chunks)

        return vector_store

    async def create_embeddings(
        self,
        connection_string: str,
        chunks: List[Document],
        table_name: str = "parent_child_embeddings",
        index_name: str = "employee_handbook_hnsw",
    ) -> VectorStore:
        """
        Creating the embeddings in vector store
        """

        # from_connection_string runs on its own background loop, which is a
        # ProactorEventLoop on Windows and breaks psycopg async. from_engine
        # without a loop runs on the caller's (selector) loop instead.
        pg_engine = PGEngine.from_engine(create_async_engine(connection_string))

        try:
            vector_size = len(await self.embedding.aembed_query("dimension probe"))
            await pg_engine.ainit_vectorstore_table(
                table_name=table_name,
                vector_size=vector_size,
            )
        except ProgrammingError:
            # Table already exists from a previous run.
            pass

        vector_store = await PGVectorStore.create(
            engine=pg_engine,
            table_name=table_name,
            embedding_service=self.embedding,
        )

        hnsw_index = HNSWIndex(m=16, ef_construction=64, name=index_name)

        await vector_store.aadd_documents(
            documents=chunks,
            # ids=[doc.metadata["child_id"] for doc in chunks],
        )

        await vector_store.aapply_vector_index(hnsw_index)

        # vector_store.

        return vector_store

    def storing_parents(self, engine: Engine, docs: List[Document]):
        """
        Storing parent chunk in postgres data
        """

        create_query = text("""
            CREATE TABLE IF NOT EXISTS parent_documents (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                content TEXT NOT NULL,
                metadata JSONB
            )
        """)

        query = text("""
            INSERT INTO parent_documents (
                id,
                document_id,
                content,
                metadata
            )
            VALUES (
                :id,
                :document_id,
                :content,
                CAST(:metadata AS JSONB)
            )
            ON CONFLICT (id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata
        """)

        with engine.begin() as connection:

            connection.execute(create_query)

            for doc in docs:

                connection.execute(
                    query,
                    {
                        "id": doc.metadata["parent_id"],
                        "document_id": doc.metadata.get("parent_id"),
                        "content": doc.page_content,
                        "metadata": json.dumps(doc.metadata),
                    },
                )

    async def get_existing_store(
        self, connection_string: str, table_name: str
    ) -> PGVectorStore:

        # from_engine has no background loop, so sync methods (create_sync,
        # similarity_search, ...) raise. Use the async API on the caller's loop.

        hybrid_search_config = build_hybrid_search_config(
            index_name=f"{table_name}_tsv_index"
        )

        pg_engine = PGEngine.from_engine(create_async_engine(connection_string))

        vector_store = await PGVectorStore.create(
            engine=pg_engine,
            table_name=table_name,
            embedding_service=self.embedding,
            hybrid_search_config=hybrid_search_config,
        )

        # create the GIN index on the tsv_column used for text search, only once.
        # the library issues a plain CREATE INDEX (no IF NOT EXISTS), so guard it.
        if not await vector_store.ais_valid_index(hybrid_search_config.index_name):
            await vector_store.aapply_hybrid_search_index()

        return vector_store

    async def storing_recursive_markdown_chunks(
        self,
        connection_string: str,
        chunks: List[Document],
        table_name: str,
        index_name: str,
    ) -> VectorStore:

        vector_store = await self.create_embeddings(
            connection_string=connection_string,
            chunks=chunks,
            table_name=table_name,
            index_name=index_name,
        )

        return vector_store
