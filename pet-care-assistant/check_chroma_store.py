from langchain_community.vectorstores import Chroma
import re

PERSIST_DIRECTORY = "data/chroma_db"

# Define keywords that should be present for relevance (customize as needed)
RELEVANT_KEYWORDS = [
    "pet", "dog", "cat", "adoption", "training", "nutrition", "grooming", "health", "safety", "behavior", "supplies", "veterinarian", "wellness", "care", "crate", "allergies", "senior", "puppy", "kitten"
]

# Load the existing ChromaDB vector store
db = Chroma(persist_directory=PERSIST_DIRECTORY)

# Fetch all documents and their ids
data = db.get()
docs = data['documents']
ids = data['ids']

relevant_docs = []
relevant_ids = []
removed_count = 0

for doc, doc_id in zip(docs, ids):
    # Check for minimum length and at least one relevant keyword
    if doc and len(doc.strip()) > 30 and any(kw in doc.lower() for kw in RELEVANT_KEYWORDS):
        relevant_docs.append(doc)
        relevant_ids.append(doc_id)
    else:
        removed_count += 1

print(f"Relevant documents kept: {len(relevant_docs)}")
print(f"Non-relevant documents removed: {removed_count}")

db.delete_collection()
db = Chroma(persist_directory=PERSIST_DIRECTORY)
from langchain_core.documents import Document
db.add_documents([Document(page_content=doc) for doc in relevant_docs])
print("Store cleaned and rebuilt with only relevant documents.")
