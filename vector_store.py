# import faiss
# import os
# import pickle
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings

# INDEX_FILE = "faiss_index"

# def load_vector_store():
#     if os.path.exists(INDEX_FILE):
#         with open(INDEX_FILE, "rb") as f:
#             return pickle.load(f)
#     else:
#         return None

# def save_vector_store(vs):
#     with open(INDEX_FILE, "wb") as f:
#         pickle.dump(vs, f)

# def create_vector_store(documents):
#     embedding = OpenAIEmbeddings()
#     vectordb = FAISS.from_documents(documents, embedding)
#     save_vector_store(vectordb)
#     return vectordb

# import faiss
# import os
# import pickle
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings  # ✅ Import HuggingFace Embeddings

# INDEX_FILE = "faiss_index"

# def load_vector_store():
#     if os.path.exists(INDEX_FILE):
#         with open(INDEX_FILE, "rb") as f:
#             return pickle.load(f)
#     else:
#         return None

# def save_vector_store(vs):
#     with open(INDEX_FILE, "wb") as f:
#         pickle.dump(vs, f)

# def create_vector_store(documents):
#     embedding = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"  # ✅ Lightweight fast model
#     )
#     vectordb = FAISS.from_documents(documents, embedding)
#     save_vector_store(vectordb)
#     return vectordb

# # app/vector_store.py

# import faiss
# import os
# import pickle
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings  # ✅ Import HuggingFace Embeddings


# INDEX_FILE = "faiss_index"

# def load_vector_store():
#     if os.path.exists(INDEX_FILE):
#         with open(INDEX_FILE, "rb") as f:
#             return pickle.load(f)
#     else:
#         return None

# def save_vector_store(vs):
#     with open(INDEX_FILE, "wb") as f:
#         pickle.dump(vs, f)

# # def create_vector_store(documents):
# #     embedding = OpenAIEmbeddings()
# #     vectordb = FAISS.from_documents(documents, embedding)
# #     save_vector_store(vectordb)
# #     return vectordb
# def create_vector_store(documents):
#     embedding = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"  # ✅ Lightweight fast model
#     )
#     vectordb = FAISS.from_documents(documents, embedding)
#     save_vector_store(vectordb)
#     return vectordb


# def add_to_vector_store(documents):
#     vectordb = load_vector_store()
#     embedding = OpenAIEmbeddings()

#     if vectordb is None:
#         vectordb = FAISS.from_documents(documents, embedding)
#     else:
#         vectordb.add_documents(documents)

#     save_vector_store(vectordb)
#     return vectordb

# app/vector_store.py

import os
import pickle
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Correct Import
from langchain_openai.embeddings import OpenAIEmbeddings
INDEX_FILE = "faiss_index"

# Create a global embedding model (so it is reused everywhere)
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",openai_api_key=openai_api_key
)

def load_vector_store():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return None

def save_vector_store(vs):
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(vs, f)

def create_vector_store(documents):
    vectordb = FAISS.from_documents(documents, embedding_model)
    save_vector_store(vectordb)
    return vectordb

def add_to_vector_store(documents):
    vectordb = load_vector_store()

    if vectordb is None:
        vectordb = FAISS.from_documents(documents, embedding_model)
    else:
        vectordb.add_documents(documents)

    save_vector_store(vectordb)
    return vectordb
