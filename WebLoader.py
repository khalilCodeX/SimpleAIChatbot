from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4

def load_website(url: str):
    #bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(url,)
        #bs_kwargs={"parse_only": bs4_strainer}
    )
    documents = loader.load()
    assert len(documents) > 0, "No documents were loaded from the website."
    print(f"Loaded {len(documents[0].page_content)} documents from {url}")

    return documents

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_splits = text_splitter.split_documents(documents)
    print(f"Split into {len(all_splits)} chunks.")

    return all_splits