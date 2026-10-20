from rag_pipeline.pre_processing.cleaning import cleaning_process
from rag_pipeline.data_parsing import PdfParser
from typing import Literal, List, Tuple
from uuid import uuid4
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class ChunkingStrategies:
    """
    Collection of different different chunking strategies
    """

    def __init__(self, dir_path: str, mode: Literal["page", "single"] = "single"):

        parser = PdfParser()

        unprocessed_doc_lists: List[List[Document]] = parser.load_documents_from_dir(
            dir_path=dir_path.resolve(), mode=mode
        )

        self.docs: List[Document] = cleaning_process(unprocessed_doc_lists)

    def markdown_recursive_chunking(self) -> List[Document]:
        """
        First do markdown chunking and then do recursive chunking
        it preserve semantic meaning and keep chunk size lows
        since pdf data resolved in the markdown format. so semantic meaning is necessary
        to be stay and chunk size no so long it is fill context window fast
        """

        # this will tell markdown splitter to what is hierarchy
        # suppose #, ## & ### in one section so it will added to the metadata

        """
        H1: Access Card Policy
        │
        ├── H2: Scope
        │
        ├── H2: Access Card Policy
        │   │
        │   ├── H3: Lost Card
        │   │
        │   └── H3: Damaged Card

        """

        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250,
        )

        header_chunks = []

        for doc in self.docs:
            chunks = markdown_splitter.split_text(doc.page_content)

            for chunk in chunks:
                chunk.metadata = {
                    **doc.metadata,
                    **chunk.metadata,
                }

            header_chunks.extend(chunks)

        final_chunks = recursive_splitter.split_documents(header_chunks)

        return final_chunks

    def parent_child_chunking(self) -> Tuple[List[Document], List[Document]]:

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "],
        )

        parent_chunks: List[Document] = []
        children_chunks: List[Document] = []

        for doc in self.docs:
            parent_docs = parent_splitter.split_documents([doc])

            for parent in parent_docs:
                parent_id = str(uuid4())

                parent.metadata = {
                    **parent.metadata,
                    **doc.metadata,
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                }

                parent_chunks.append(parent)

                child_docs = child_splitter.split_documents([parent])

                for child in child_docs:

                    child.metadata = {
                        **child.metadata,
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "child_id": str(uuid4()),
                    }

                    children_chunks.append(child)

        return parent_chunks, children_chunks
