# from dotenv import load_dotenv
# import os

# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAI
# from document_loader import extract_text

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # Initialize the LLM (using OpenAI)
# llm = OpenAI(model = "gpt-4o",openai_api_key=openai_api_key)

# # Function to set up the RAG system
# def setup_rag_system():
#     # Load the document
#     # loader = TextLoader('data/my_document.txt')
#     # documents = loader.load()
#     documents = extract_text(mode="local")

#     # Split the document into chunks
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     document_chunks = splitter.split_documents(documents)

#     # Initialize embeddings with OpenAI API key
#     #embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Correct Import

#     embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

#     # Create FAISS vector store from document chunks and embeddings
#     vector_store = FAISS.from_documents(document_chunks, embeddings)

#     # Return the retriever for document retrieval with specified search_type
#     retriever = vector_store.as_retriever(
#         search_type="similarity",  # or "mmr" or "similarity_score_threshold"
#         search_kwargs={"k": 5}  # Adjust the number of results if needed
#     )
#     return retriever

# # Function to get the response from the RAG system
# async def get_rag_response(query: str):
#     retriever = setup_rag_system()

#     # Retrieve the relevant documents using 'get_relevant_documents' method
#     retrieved_docs = retriever.invoke(query)

#     # Prepare the input for the LLM: Combine the query and the retrieved documents into a single string
#     context = "\n".join([doc.page_content for doc in retrieved_docs])

#     # LLM expects a list of strings (prompts), so we create one by combining the query with the retrieved context
#     prompt = [f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}"]

#     # Generate the final response using the language model (LLM)
#     generated_response = llm.generate(prompt)  # Pass as a list of strings
    
#     return generated_response


# rag_pipeline.py

# from dotenv import load_dotenv
# import os

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document

# from .document_loader import extract_text

# # Load environment variables
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

# # Initialize LLM and Embeddings
# llm = OpenAI(model="gpt-4o", openai_api_key=openai_api_key)
# # from langchain_openai import ChatOpenAI
# # llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # -------------------------------
# # Functions
# # -------------------------------

# # def load_and_split_documents(file_path: str) -> list:
# #     """
# #     Load raw document text and split into chunks manually.
# #     """
# #     # Step 1: Extract raw text
# #     pages = extract_text(mode="local", file_path=file_path)

# #     # Step 2: Split raw text
# #     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# #     chunks = splitter.split_text(pages)

# #     # Step 3: Wrap each chunk as a Document
# #     chunk_documents = [Document(page_content=chunk) for chunk in chunks]

# #     return chunk_documents

# def load_and_split_documents(file_bytes: bytes) -> list:
#     """
#     Load raw document text from bytes and split into chunks manually.
#     """
#     pages = extract_text(mode="memory", file_bytes=file_bytes)

#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_text(pages)

#     chunk_documents = [Document(page_content=chunk) for chunk in chunks]

#     return chunk_documents


# def create_vectorstore(chunk_documents: list) -> FAISS:
#     """
#     Create FAISS vectorstore from chunked documents.
#     """
#     vector_store = FAISS.from_documents(chunk_documents, embeddings)
#     return vector_store

# def create_retriever(vector_store: FAISS):
#     """
#     Create retriever from FAISS vectorstore.
#     """
#     retriever = vector_store.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 5}
#     )
#     return retriever

# async def generate_rag_response(query: str, retriever) -> str:
#     """
#     Use retriever and LLM to generate RAG response.
#     """
#     retrieved_docs = retriever.invoke(query)

#     # Prepare context
#     context = "\n".join([doc.page_content for doc in retrieved_docs])

#     # Prepare prompt
#     prompt = [f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"]

#     # Generate response
#     generated_response = llm.generate(prompt)

#     return generated_response

# # -------------------------------
# # (Optional) Full utility
# # -------------------------------

# async def full_rag_pipeline(file_path: str, query: str):
#     """
#     Full pipeline: load -> split -> vectorize -> retrieve -> answer.
#     """
#     chunk_documents = load_and_split_documents(file_path)
#     vector_store = create_vectorstore(chunk_documents)
#     retriever = create_retriever(vector_store)
#     response = await generate_rag_response(query, retriever)
#     return response

from dotenv import load_dotenv
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from document_loader import extract_text

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY:", openai_api_key)

# Initialize LLM and Embeddings
#llm = OpenAI(model="gpt-4o", openai_api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
print("LLM:", llm)

#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",openai_api_key=openai_api_key
)

# -------------------------------
# Functions
# -------------------------------

def load_and_split_documents(file_bytes: bytes) -> list:
    """
    Load raw document text from bytes and split into chunks manually.
    """
    pages = extract_text(mode="memory", file_bytes=file_bytes)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(pages)

    chunk_documents = [Document(page_content=chunk) for chunk in chunks]

    return chunk_documents

def create_vectorstore(chunk_documents: list) -> FAISS:
    """
    Create FAISS vectorstore from chunked documents.
    """
    vector_store = FAISS.from_documents(chunk_documents, embeddings)
    return vector_store

def create_retriever(vector_store: FAISS):
    """
    Create retriever from FAISS vectorstore.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    return retriever

def generate_rag_response(query: str, retriever) -> str:
    """
    Use retriever and LLM to generate RAG response.
    """
    retrieved_docs = retriever.invoke(query)

    # Prepare context
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    print("TYPE OF CONTEXT:", type(context))
    print("CONTEXT VALUE:", context)

    # Prepare prompt
    prompt = [f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"]

    # Generate response
    generated_response = llm.invoke(prompt)
    print("TYPE OF RESPONSE1:", type(generated_response))
    print("RESPONSE VALUE1:", generated_response)

    # Extract pure text
    answer = generated_response.content if hasattr(generated_response, "content") else generated_response
    print("TYPE OF ANSWER:", type(answer))
    
    print("RESPONSE VALUE2:", answer)


    return answer.strip()

# -------------------------------
# (Optional) Full utility
# -------------------------------

def full_rag_pipeline(file_bytes: bytes, query: str):
    """
    Full pipeline: load -> split -> vectorize -> retrieve -> answer.
    """
    chunk_documents = load_and_split_documents(file_bytes)
    vector_store = create_vectorstore(chunk_documents)
    retriever = create_retriever(vector_store)
    response = generate_rag_response(query, retriever)
    return response













