"""
ingest.py

Document ingestion pipeline for building a vector database.

This script:
1. Loads documents from multiple file formats
2. Splits them into chunks
3. Converts chunks into embeddings
4. Stores them in a persistent Chroma vector database

Supports parallel document loading and avoids re-ingesting files
already present in the vector store.
"""

import os
import glob
import shutil
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


# =====================================================
# Configuration (Environment Variables)
# =====================================================

# Directory where Chroma DB will be stored
persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db')

# Delete old vector DB (fresh ingestion every run)
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# Directory containing raw source documents
source_directory = os.environ.get('SOURCE_DIRECTORY', 'raw_txt')

# Embedding model for converting text → vectors
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')

# Text chunking settings
chunk_size = 1000        # Max tokens per chunk
chunk_overlap = 50       # Overlap between chunks for context retention


# =====================================================
# Custom Loader for Emails
# =====================================================

class MyElmLoader(UnstructuredEmailLoader):
    """
    Custom email loader that falls back to 'text/plain'
    if HTML content is not found.
    """

    def load(self) -> List[Document]:
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                # Some emails don't contain HTML body
                if 'text/html content not found in email' in str(e):
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file path context to the error
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# =====================================================
# File Type → Loader Mapping
# =====================================================

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


# =====================================================
# Document Loading Functions
# =====================================================

def load_single_document(file_path: str) -> List[Document]:
    """
    Loads a single document based on its file extension.

    Args:
        file_path (str): Path to the document file.

    Returns:
        List[Document]: Loaded document objects.
    """
    ext = "." + file_path.rsplit(".", 1)[-1]

    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all supported documents from a directory using multiprocessing.

    Args:
        source_dir (str): Folder containing documents.
        ignored_files (List[str]): Files already in vector DB.

    Returns:
        List[Document]: All loaded documents.
    """
    all_files = []

    # Find all supported file types
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )

    # Remove already-ingested files
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    # Parallel document loading
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for docs in pool.imap_unordered(load_single_document, filtered_files):
                results.extend(docs)
                pbar.update()

    return results


# =====================================================
# Text Processing
# =====================================================

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Loads documents and splits them into chunks.

    Args:
        ignored_files (List[str]): Files to skip.

    Returns:
        List[Document]: Chunked documents ready for embeddings.
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)

    if not documents:
        print("No new documents to load")
        exit(0)

    print(f"Loaded {len(documents)} new documents from {source_directory}")

    # Split documents into chunks for better retrieval
    text_splitter = SpacyTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)

    print(f"Split into {len(texts)} chunks of text (max {chunk_size} tokens each)")
    return texts


# =====================================================
# Vector Store Utilities
# =====================================================

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks whether a valid Chroma vector store exists.

    Returns:
        bool: True if vector store exists.
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) \
           and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            if len(list_index_files) > 3:
                return True
    return False


# =====================================================
# Main Ingestion Pipeline
# =====================================================

def main():
    """
    Entry point for ingestion.

    - Creates embeddings
    - Loads + processes documents
    - Builds or updates vector DB
    """
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        collection = db.get()

        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])

        print("Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        print("Creating new vectorstore")
        texts = process_documents()

        print("Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

    db.persist()
    db = None

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
