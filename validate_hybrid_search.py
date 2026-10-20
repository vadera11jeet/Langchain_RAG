"""
Check that hybrid search (dense + full-text, fused with RRF) is actually running.

For each query it prints:
  - dense-only top-k          (hybrid_search_config=None)
  - keyword-only top-k        (raw full-text query, same SQL the library uses)
  - hybrid top-k              (what Retriever.similarity_search returns)
and tags each hybrid hit with D (found by dense) / K (found by keyword).
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from rag_pipeline.retrieval.retriever import Retriever
from rag_pipeline.vectorstore.vector_store import VectorStore

TABLE = "markdown_recursive_chunks"
LANG = "pg_catalog.english"
K = 10

QUERIES = [
    "Can you tell me about Alcohol and Substance Abuse policy ?",
    "can you tell me about employee Referral Bonus Eligibility? ",
    # plainto_tsquery ANDs every non-stopword term, so filler like "tell" kills
    # keyword matches; keyword-style queries show the sparse side working.
    "Alcohol and Substance Abuse policy",
    "employee Referral Bonus Eligibility",
]


def short(content: str, n: int = 70) -> str:
    return content.replace("\n", " ")[:n]


async def check_index(engine) -> None:
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = :t"),
                {"t": TABLE},
            )
        ).all()
    print("== indexes on", TABLE)
    for name, definition in rows:
        print(f"  {name}: {definition}")
    if not any("to_tsvector" in d for _, d in rows):
        print("  !! no full-text GIN index found")


async def keyword_search(engine, query: str) -> list[tuple[str, str, float]]:
    sql = text(f"""
        SELECT langchain_id::text, content,
               ts_rank_cd(to_tsvector('{LANG}', content), plainto_tsquery('{LANG}', :q)) AS rank
        FROM "{TABLE}"
        WHERE to_tsvector('{LANG}', content) @@ plainto_tsquery('{LANG}', :q)
        ORDER BY rank DESC
        LIMIT :k
    """)
    async with engine.connect() as conn:
        tsquery = (
            await conn.execute(text(f"SELECT plainto_tsquery('{LANG}', :q)::text"), {"q": query})
        ).scalar()
        rows = (await conn.execute(sql, {"q": query, "k": K})).all()
    print(f"  tsquery: {tsquery}")
    return [(r[0], r[1], r[2]) for r in rows]


async def main():
    connection_url = os.environ["VECTOR_STORE_CONNECTION_STRING"]
    engine = create_async_engine(connection_url)

    store = await VectorStore().get_existing_store(
        connection_string=connection_url, table_name=TABLE
    )
    retriever = Retriever(store)

    await check_index(engine)

    for query in QUERIES:
        print("\n" + "=" * 100)
        print("QUERY:", query)

        dense = await store.asimilarity_search_with_score(
            query, k=K, hybrid_search_config=None
        )
        print(f"\n-- dense only ({len(dense)}) | score = cosine distance, lower is better")
        for doc, score in dense:
            print(f"  {score:.4f}  {doc.id}  {short(doc.page_content)}")

        print("\n-- keyword only | score = ts_rank_cd, higher is better")
        keyword = await keyword_search(engine, query)
        print(f"  {len(keyword)} rows matched")
        for doc_id, content, rank in keyword:
            print(f"  {rank:.4f}  {doc_id}  {short(content)}")

        hybrid = await retriever.similarity_search(query, K)
        dense_ids = {str(d.id) for d, _ in dense}
        keyword_ids = {doc_id for doc_id, _, _ in keyword}
        print(f"\n-- hybrid ({len(hybrid)}) | score = RRF, higher is better (max ~{2 / 60:.4f})")
        for doc, score in hybrid:
            tag = ("D" if str(doc.id) in dense_ids else "-") + (
                "K" if str(doc.id) in keyword_ids else "-"
            )
            print(f"  {score:.4f}  [{tag}]  {doc.id}  {short(doc.page_content)}")

        only_keyword = [d for d, _ in hybrid if str(d.id) in keyword_ids - dense_ids]
        print(
            f"\n  hybrid hits from keyword only (not in dense top-{K}): {len(only_keyword)}"
        )

    await engine.dispose()


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
    else:
        asyncio.run(main())
