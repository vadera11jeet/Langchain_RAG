from rag_pipeline.chunking.Chunking import ChunkingStrategies
from pathlib import Path


def main():
    BASE_DIR = Path(__file__).resolve().parent
    chunking = ChunkingStrategies(dir_path=BASE_DIR / "docs", mode="single")

    chunks = chunking.markdown_recursive_chunking()

    print(chunks)


if __name__ == "__main__":
    main()
