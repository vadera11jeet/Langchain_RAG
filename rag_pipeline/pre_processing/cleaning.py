from typing import List
from langchain_core.documents import Document
from rag_pipeline.pre_processing.pre_processor import (
    extract_version,
    remove_footer,
    remove_page_marker,
    normalize_whitespace,
)


def cleaning_process(doc_lists: List[List[Document]] = []) -> List[Document]:
    """
    Cleaning the pdf removing the headers, footer and page number
    adding version in metadata
    """

    pre_processed_documents: List[Document] = []

    for docs in doc_lists:
        version = None

        for doc in docs:
            content = doc.page_content
            # extracting version from the chunks
            version = version or extract_version(content)

            # removing the footer text
            content = remove_footer(content)

            # remove the page watermark
            content = remove_page_marker(content)

            # removing the extra white space and unwanted \n
            content = normalize_whitespace(content)

            doc.page_content = content

            doc.metadata = {**doc.metadata, "version": version if version else "1.0"}

            pre_processed_documents.append(doc)

    return pre_processed_documents
