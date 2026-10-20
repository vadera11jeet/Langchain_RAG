from rag_pipeline.chunking.Chunking import ChunkingStrategies
from rag_pipeline.vectorstore.vector_store import VectorStore
from rag_pipeline.retrieval.retriever import Retriever
from rag_pipeline.generation.llm import LLM, LLMResponse
from rag_pipeline.reranker.reranker import ReRanker
from sqlalchemy import create_engine
from pathlib import Path
from typing import Literal, List, Tuple
from langchain_core.documents import Document
import os


class RAG:

    connection_url: str = os.environ["VECTOR_STORE_CONNECTION_STRING"]
    engine = create_engine(connection_url)

    def __init__(self):
        pass

    async def parent_child_chunking(
        self, path: str, mode: Literal["single", "page"]
    ) -> Tuple[List[Document], List[Document]]:

        chunking = ChunkingStrategies(dir_path=path, mode=mode)

        parent_chunks, children_chunks = chunking.parent_child_chunking()

        return parent_chunks, children_chunks

    async def create_parent_child_embedding(
        self,
        parent_chunks: List[Document],
        children_chunks: List[Document],
        collection_name: str,
        table_name: str = "parent_child_chunking",
        index_name: str = "parent_child_chunking_index",
    ) -> VectorStore:
        vector_store = VectorStore(collection_name=collection_name)

        await vector_store.storing_parent_child_embeddings(
            engine=RAG.engine,
            parent_chunks=parent_chunks,
            children_chunks=children_chunks,
            connection_string=RAG.connection_url,
            table_name=table_name,
            index_name=index_name,
        )

        # return the wrapper, retriever_parent_child calls get_existing_store on it
        return vector_store

    async def retriever_parent_child(
        self, table_name: str, vector_store: VectorStore, query: str, top_k: int
    ) -> List[Tuple[str, str]]:
        store = await vector_store.get_existing_store(
            connection_string=RAG.connection_url, table_name=table_name
        )

        retriever = Retriever(store)

        child_docs = await retriever.similarity_search(query, top_k)

        parent_ids = list({child.metadata["parent_id"] for child, _ in child_docs})

        parent_content = await retriever.parent_retrieval(
            engine=RAG.engine, parent_ids=parent_ids
        )

        return parent_content

    async def parent_child_generation(
        self, docs: List[Tuple[str, str]], query: str, rerank_top_k: int
    ) -> LLMResponse:

        llm = LLM()
        # reranker = ReRanker()

        # rank_docs = reranker.cross_encoding_re_ranker(docs, query, rerank_top_k)

        response = await llm.invoke_parent_child(query, docs)

        return response

    async def parent_child_RAG(
        self,
        query: str,
        collection_name: str = "collection_1",
        table_name: str = "parent_child_embeddings",
        top_k: int = 25,
        rerank_top_k: int = 7,
    ) -> LLMResponse:
        vector_store = VectorStore(collection_name=collection_name)

        parent_docs = await self.retriever_parent_child(
            table_name=table_name,
            vector_store=vector_store,
            query=query,
            top_k=top_k,
        )

        response = await self.parent_child_generation(
            docs=parent_docs, query=query, rerank_top_k=rerank_top_k
        )

        return response

    async def markdown_recursive_chunking(
        self, path: str, mode: Literal["single", "page"]
    ) -> List[Document]:

        chunking = ChunkingStrategies(dir_path=path, mode=mode)

        chunks = chunking.markdown_recursive_chunking()

        return chunks

    async def create_markdown_recursive_embeddings(
        self,
        chunks: List[Document],
        collection_name: str,
        table_name: str = "markdown_recursive_chunks",
        index_name: str = "markdown_recursive_index",
    ) -> VectorStore:
        vector_store = VectorStore(collection_name=collection_name)

        await vector_store.storing_recursive_markdown_chunks(
            connection_string=RAG.connection_url,
            chunks=chunks,
            table_name=table_name,
            index_name=index_name,
        )

        # return the wrapper, retriever_markdown_recursive calls get_existing_store on it
        return vector_store

    async def retriever_markdown_recursive(
        self,
        vector_store: VectorStore,
        query: str,
        top_k: int,
        table_name: str = "markdown_recursive_chunks",
    ) -> List[Tuple[Document, float]]:
        store = await vector_store.get_existing_store(
            connection_string=RAG.connection_url, table_name=table_name
        )

        retriever = Retriever(store)

        docs = await retriever.similarity_search(query, top_k)

        return docs

    async def markdown_recursive_generation(
        self, docs: List[Tuple[Document, float]], query: str, rerank_top_k: int
    ) -> LLMResponse:

        llm = LLM()
        reranker = ReRanker()

        rank_docs = reranker.cross_encoding_re_ranker(docs, query, rerank_top_k)

        response = llm.invoke(query, rank_docs)

        return response
