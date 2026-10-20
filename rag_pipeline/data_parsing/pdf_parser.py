from dotenv import load_dotenv

load_dotenv()

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.documents.base import Document
from langchain_core.documents import Document
from typing import Literal, Union, Iterator
from pathlib import Path


class PdfParser:

    def __str__(self):
        print("Pdf parser class")

    def load_documents(
        self,
        file_path: str,
        mode: Literal["page", "single"] = "single",
        extract_images: bool = False,
        lazy_load: bool = True,
    ) -> Union[list[list[Document]], Iterator[Document]]:

        try:

            pdf_loader = PyMuPDF4LLMLoader(
                file_path=file_path,
                mode=mode,
                extract_images=extract_images,
            )

            if lazy_load:
                return pdf_loader.lazy_load()

            return pdf_loader.load()

        except Exception as e:
            print(f"failed to parse {file_path}: {e}")
            return None

    def load_documents_from_dir(
        self,
        dir_path: str = "./",
        lazy_load: bool = False,
        mode: Literal["page", "single"] = "single",
        extract_images: bool = False,
    ) -> Union[list[Document], Iterator[Document]]:

        folder_path = Path(dir_path)

        pdf_files = list(folder_path.glob("*.pdf"))
        docs = []

        for pdf_file_path in pdf_files:
            doc = self.load_documents(
                pdf_file_path,
                lazy_load=lazy_load,
                mode=mode,
                extract_images=extract_images,
            )
            if doc is None:
                continue
            docs.append(doc)

        return docs


if __name__ == "__main__":
    parser = PdfParser()
    doc_lists: list[list[Document]] = parser.load_documents_from_dir("../../docs")

    for number, docs in enumerate(doc_lists):
        for i, doc in enumerate(docs):
            with open(f"{number+1}.txt", "a", encoding="utf-8") as file:
                file.write(doc.page_content)
