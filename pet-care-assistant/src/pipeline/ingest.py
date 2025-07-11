import os
from dotenv import load_dotenv
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables from .env file
load_dotenv()

# Directory for storing the vector database
PERSIST_DIRECTORY = "data/chroma_db"

# Debug: Print loaded Tavily API key
api_key_debug = os.getenv("TAVILY_API_KEY")
print("Loaded Tavily API key:", repr(api_key_debug))

def get_broad_pet_life_urls(num_results_per_query=5):
    """
    Fetch URLs covering the most essential aspects of pet ownership using Tavily via LangChain tool.
    Only include URLs from trusted domains.
    """
    queries = [
        "pet care basics",
        "pet adoption tips",
        "pet training techniques",
        "pet health care",
        "pet nutrition advice",
        "pet grooming tips",
        "pet safety tips",
        "pet first aid",
        "pet behavior",
        "pet supplies",
        "pet vaccination schedule",
        "pet dental care",
        "pet exercise routines",
        "pet mental stimulation",
        "pet travel tips",
        "pet insurance",
        "pet emergency preparedness",
        "pet socialization",
        "pet senior care",
        "pet toxic foods",
        "pet parasite prevention",
        "pet adoption process",
        "pet crate training",
        "pet separation anxiety",
        "pet allergies",
        "pet cleaning tips"
    ]
    allowed_domains = [
        "akc.org", "aspca.org", "humanesociety.org", "petmd.com", "petfinder.com",
        ".edu", ".gov"
    ]
    tavily_tool = TavilySearchResults()
    urls = set()
    for query in queries:
        results = tavily_tool.run({"query": query, "max_results": num_results_per_query})
        # TavilySearchResults returns a list of dicts with 'url' keys
        for item in results:
            url = item.get('url')
            if url and any(domain in url for domain in allowed_domains):
                urls.add(url)
            if len(urls) >= num_results_per_query * len(queries):
                break
    return list(urls)

def main():
    """
    Main function to load data, split it, and store it in a vector database.
    """
    print("--- Starting Data Ingestion ---")

    # 1. Discover Broad Pet Life URLs
    print("Fetching broad pet life URLs using Tavily API...")
    urls = get_broad_pet_life_urls(num_results_per_query=5)
    print(f"Found {len(urls)} unique URLs.")

    all_docs = []
    for url in urls:
        print(f"Loading documents from {url}...")
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            print(f"Loaded {len(docs)} document(s) from {url}.")
            all_docs.extend(docs)
        except Exception as e:
            print(f"Failed to load {url}: {e}")

    if not all_docs:
        print("No documents loaded. Exiting.")
        return

    # 2. Split Documents
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_docs)
    print(f"Split into {len(splits)} chunks.")

    # 3. Create Embeddings and Store in ChromaDB
    print(f"Creating vector store at {PERSIST_DIRECTORY}...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("--- Data Ingestion Complete ---")
    print(f"Total vectors in store: {vectorstore._collection.count()}")

if __name__ == "__main__":
    main()
