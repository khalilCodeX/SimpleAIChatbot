from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

def query_vectorstore(retriever, query: str):
    return retriever.get_relevant_documents(query)