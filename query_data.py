#!/usr/bin/env python
import argparse

from langchain_community.llms.ollama import Ollama
import logs
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from logs import log, logd
from utils import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Here is a context:
```
{context}
```

Answer the following question based on the above context. Be succint and clear:
```
{question}
```
"""


def main():
    parser = argparse.ArgumentParser(description=f"Query the database '{CHROMA_PATH}'")
    parser.add_argument("--reset", action="store_true", help="reset the db")
    parser.add_argument("--debug", action="store_true", help="print debug info")
    parser.add_argument("query", help="the query string")
    args = parser.parse_args()
    if args.debug:
        logs.debug = True
    query_rag(args.query)


def query_rag(query: str):
    embeddings = get_embedding_function()
    log(f"Query: {query}")
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    sources = db.similarity_search_with_score(query, k=5)
    contexts = []
    source_names = []
    for source, score in sources:
        logd(f"Result {source.metadata['chunk_id']} score: {format(score, '.2f')}")
        contexts.append(source.page_content)
        source_names.append(source.metadata["chunk_id"])
    context = "\n\n--\n\n".join(contexts)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query)
    logd(f"Prompt: {prompt}")

    model = Ollama(model="mistral")
    response = model.invoke(prompt)
    log(f"Response: {response.strip()}")
    log(f"Sources: {source_names}")


if __name__ == "__main__":
    main()
