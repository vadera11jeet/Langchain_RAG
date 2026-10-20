
--- DDL configuration ---

--- Creating the vector store
CREATE EXTENSION IF NOT EXISTS vector;

--- Creating the table for storing metadata
CREATE TABLE parent_documents (
    id UUID PRIMARY KEY,
    document_id TEXT,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

--- Check which type of index has been used on the table

SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'langchain_pg_embedding';


--- creating the HNSW index not required only done by the langchain
CREATE INDEX embedding_hnsw_idx
ON langchain_pg_embedding
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,
    ef_construction = 100
);

--- use this query to verify which type of index is existing
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'parent_child_embeddings';