from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from typing import Optional

from rag_pipeline import (
    load_and_split_documents,
    create_vectorstore,
    create_retriever,
    generate_rag_response
)

app = FastAPI()

# @app.post("/rag", response_class=PlainTextResponse)
# async def rag_api(
#     query: str = Form(...),
#     file: UploadFile = File(...)
# ):
#     try:
#         # Step 1: Read uploaded file bytes
#         file_bytes = await file.read()

#         # Step 2: Load and split documents
#         documents = load_and_split_documents(file_bytes=file_bytes)

#         # Step 3: Create vectorstore
#         vectorstore = create_vectorstore(documents)

#         # Step 4: Create retriever
#         retriever = create_retriever(vectorstore)

#         # Step 5: Generate response from RAG
#         response = generate_rag_response(query=query, retriever=retriever)

#         # Step 6: Return plain text response
#         return response

#     except Exception as e:
#         return PlainTextResponse(content=f"Error: {str(e)}", status_code=500)

@app.post("/rag", response_class=PlainTextResponse)
async def rag_api(
    query: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        print("Step 1: Reading file...")
        file_bytes = await file.read()

        print("Step 2: Splitting document...")
        documents = load_and_split_documents(file_bytes=file_bytes)
        print(f"Loaded {len(documents)} documents")

        print("Step 3: Creating vectorstore...")
        vectorstore = create_vectorstore(documents)

        print("Step 4: Creating retriever...")
        retriever = create_retriever(vectorstore)

        print("Step 5: Generating response...")
        response = generate_rag_response(query=query, retriever=retriever)
        print("Raw response:", response)
        print("Response type:", type(response))

        if hasattr(response, "content"):
            return response.content
        return str(response)

    except Exception as e:
        print("Exception:", str(e))
        return PlainTextResponse(content=f"Error: {str(e)}", status_code=500)

