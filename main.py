import asyncio
from dotenv import load_dotenv

load_dotenv()
from rag_pipeline.chunking.Chunking import ChunkingStrategies
from rag_pipeline.vectorstore.vector_store import VectorStore
from rag_pipeline.retrieval.retriever import Retriever
from rag_pipeline.generation.llm import LLMResponse
from rag_pipeline.reranker.reranker import ReRanker
from rag_pipeline.rag.RAG import RAG
from pathlib import Path
from sqlalchemy import create_engine
import os


async def parent_child_rag(is_embedding_required: bool = False) -> LLMResponse:
    parent_child_rag = RAG()
    query = "Can you tell me about Alcohol and Substance Abuse policy ?"
    vector_store = VectorStore()

    if is_embedding_required:
        BASE_DIR = Path(__file__).resolve().parent

        parent_chunks, child_chunks = await parent_child_rag.parent_child_chunking(
            BASE_DIR / "docs", "single"
        )

        vector_store = await parent_child_rag.create_parent_child_embedding(
            parent_chunks=parent_chunks,
            children_chunks=child_chunks,
            collection_name="parent_child_chunks",
            table_name="parent_child_embedding",
            index_name="HNSW_parent_child_embedding_index",
        )

    content = await parent_child_rag.retriever_parent_child(
        table_name="parent_child_embedding",
        query=query,
        vector_store=vector_store,
        top_k=10,
    )

    ans = await parent_child_rag.parent_child_generation(
        docs=content, query=query, rerank_top_k=7
    )

    return ans


async def markdown_recursive_rag(is_embedding_required: bool = False) -> LLMResponse:
    markdown_recursive_rag = RAG()
    query = "Can you tell me about Alcohol and Substance Abuse policy ?"
    # same table for writing and reading the embeddings
    table_name = "markdown_recursive_chunks"
    vector_store = VectorStore()
    if is_embedding_required:
        BASE_DIR = Path(__file__).resolve().parent

        chunks = await markdown_recursive_rag.markdown_recursive_chunking(
            BASE_DIR / "docs", "single"
        )

        vector_store = (
            await markdown_recursive_rag.create_markdown_recursive_embeddings(
                chunks=chunks,
                collection_name="markdown_recursive_chunks",
                table_name=table_name,
                index_name="create_markdown_recursive_index",
            )
        )

    docs = await markdown_recursive_rag.retriever_markdown_recursive(
        table_name=table_name,
        query=query,
        vector_store=vector_store,
        top_k=10,
    )
    ans = await markdown_recursive_rag.markdown_recursive_generation(
        docs=docs, query=query, rerank_top_k=7
    )

    return ans


async def main():

    parent_child_ans = await parent_child_rag()

    print("parent child chunking answer")
    print(parent_child_ans["answer"])
    print()

    # set True only to (re)build the markdown_recursive_chunks table;
    # each True run appends another full copy of the chunks
    recursive_markdown_chunking_ans = await markdown_recursive_rag()

    print("markdown chunking answer")
    print(recursive_markdown_chunking_ans["answer"])

    # # chunking = ChunkingStrategies(dir_path=BASE_DIR / "docs", mode="single")

    # # chunks = chunking.markdown_recursive_chunking()

    # # parent_chunks, children_chunks = chunking.parent_child_chunking()

    # connection_url = os.environ["VECTOR_STORE_CONNECTION_STRING"]

    # engine = create_engine(connection_url)

    # vector_store = VectorStore()
    # # vector_store = VectorStore(collection_name="Recursive_markdown_chunking")

    # # store = await vector_store.storing_parent_child_embeddings(
    # #     engine=engine,
    # #     parent_chunks=parent_chunks,
    # #     children_chunks=children_chunks,
    # #     connection_string=connection_url,
    # # )

    # # store = await vector_store.storing_recursive_markdown_chunks(
    # #     connection_string=connection_url, chunks=chunks
    # # )

    # # Parent child chunking

    # # store = await vector_store.get_existing_store(
    # #     connection_string=connection_url, table_name="parent_child_embeddings"
    # # )

    # # recursive markdown chunking

    # store = await vector_store.get_existing_store(
    #     connection_string=connection_url, table_name="markdown_recursive_chunks"
    # )

    # retriever = Retriever(store)

    # # query = "can you tell me about employee Referral Bonus Eligibility? "

    # # parent_content, child_docs = await retriever.parent_child_retrieval(query, engine)

    # docs = await retriever.similarity_search(query, 10)

    # llm = LLM()

    # # response = await llm.invoke(query, parent_content)

    # reranker = ReRanker()

    # rank_docs = reranker.cross_encoding_re_ranker(docs, query, 7)
    # print("=========")
    # print("Printing re-ranking documents")
    # print(rank_docs)

    # response = llm.invoke(query, rank_docs)

    # print("==== Printing the last response  =====")
    # print(response["answer"])
    # # print("===== Printing the chunks ============")
    # # print(response["parents"])


if __name__ == "__main__":
    if os.name == "nt":
        # Psycopg's async connections require a selector-based loop on Windows.
        asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
    else:
        asyncio.run(main())
