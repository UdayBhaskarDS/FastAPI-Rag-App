# debug.py
from rag_pipeline import (
    load_and_split_documents,
    create_vectorstore,
    create_retriever,
    generate_rag_response
)

with open("data/sample.pdf", "rb") as f:
    file_bytes = f.read()

query = "Summarize the case details in 2 lines"

docs = load_and_split_documents(file_bytes)
vs = create_vectorstore(docs)
retriever = create_retriever(vs)
response = generate_rag_response(query, retriever)

print("RAG Response:\n", response)
