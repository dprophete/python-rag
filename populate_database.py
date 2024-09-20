#!/usr/bin/env python
import argparse
import glob
import os
import shutil

import logs
from langchain.document_loaders.pdf import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders.epub import UnstructuredEPubLoader
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logs import log, logd
from utils import get_embedding_function
from pypandoc.pandoc_download import download_pandoc
import nltk

# nltk.download("punkt")
# nltk.download("punkt_tab")

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    parser = argparse.ArgumentParser(
        description=f"Populate the database '{CHROMA_PATH}' with the data files from the dir '{DATA_PATH}'"
    )
    parser.add_argument("--reset", action="store_true", help="reset the db")
    parser.add_argument("--debug", action="store_true", help="print debug info")
    args = parser.parse_args()
    if args.debug:
        logs.debug = True
    if args.reset:
        reset_db()

    docs = load_documents()
    log(f"Total {len(docs)} documents")
    chunks = split_documents(docs)
    log(f"Created {len(chunks)} chunks")
    add_to_chroma(chunks)


def reset_db():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


# Load a single document
def load_pdf(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    docs = loader.load()
    logd(f"Loaded {len(docs)} pages for {path}")
    return docs


def load_epub(path: str) -> list[Document]:
    loader = UnstructuredEPubLoader(path)
    docs = loader.load()
    logd(f"Loaded {len(docs)} pages for {path}")
    for i, doc in enumerate(docs):
        doc.metadata["page"] = i
    return docs


# Load all the documents
def load_documents() -> list[Document]:
    # note: we could go document by document, but for simplicity we load all at once
    # loader = PyPDFDirectoryLoader(DATA_PATH)
    # docs = loader.load()
    docs = []
    for file in glob.glob(os.path.join(DATA_PATH, "*.pdf")):
        docs += load_pdf(file)
    for file in glob.glob(os.path.join(DATA_PATH, "*.epub")):
        docs += load_epub(file)
    return docs


def tag_chunks_for_doc(chunks: list[Document]) -> list[Document]:
    for i, chunk in enumerate(chunks):
        source = chunk.metadata["source"]
        page = chunk.metadata["page"]
        chunk.metadata["chunk_id"] = f"{source}:{page}:{i}"
    return chunks


def split_document(
    splitter: RecursiveCharacterTextSplitter, doc: Document
) -> list[Document]:
    chunks = splitter.split_documents([doc])
    chunks = tag_chunks_for_doc(chunks)
    logd(
        f"Created {len(chunks)} chunks for {doc.metadata['source']}, page {doc.metadata['page']}"
    )
    return chunks


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = []
    for doc in docs:
        chunks += split_document(splitter, doc)
    return chunks


def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    new_chunks = []
    new_ids = []
    for chunk in chunks:
        chunk_id = chunk.metadata["chunk_id"]
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if len(new_chunks) == 0:
        log("No new items to add")
    else:
        log(f"Adding {len(new_chunks)} new items")
        db.add_documents(new_chunks, ids=new_ids)


if __name__ == "__main__":
    main()
