# src/agent/nodes.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure the .env file is loaded from the correct path
try:
    env_path = Path(__file__).parent.parent.parent / '.env'
    print(f"Attempting to load .env file from: {env_path}")
    load_dotenv(dotenv_path=env_path)
except Exception as e:
    print(f"Error loading .env file: {e}")

# --- Load the Google API key ---
google_api_key = os.getenv("GOOGLE_API_KEY")
print(f"--- Loaded GOOGLE_API_KEY: '{google_api_key[:5]}...' ---") # Prints the first 5 chars

# If the key is still not found, stop with a clear error
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please check your .env file and its path.")

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

# --- Initialize LangChain Components Explicitly ---
print("Initializing Google AI components with the loaded key...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=google_api_key)
grading_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=google_api_key)

# --- Initialize Vector Store and Search Tool ---
vectorstore = Chroma(persist_directory="./data/chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()
web_search_tool = TavilySearchResults(k=3)


# --- Node Functions ---

def retrieve(state):
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    documents = retriever.invoke(question)
    print(f"---FOUND {len(documents)} DOCUMENTS---")
    return {"documents": documents, "question": question}

def grade_documents(state):
    print("---CHECKING DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keywords related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant.
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

        Here is the retrieved document:\n ------- \n{document}\n ------- \nHere is the user question: {question}""",
        input_variables=["question", "document"],
    )
    grading_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    structured_llm_grader = grading_llm.with_structured_output({
        "name": "grade_document",
        "description": "Grades the relevance of a document to a user question.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Whether the document is relevant to the user question."
                }
            },
            "required": ["score"]
        }
    })
    
    web_search_needed = False
    filtered_docs = []
    for d in documents:
        prompt_val = prompt.invoke({"question": question, "document": d.page_content})
        grade = structured_llm_grader.invoke(prompt_val)
        if grade["score"].lower() == "yes":
            print("---GRADE: DOCUMENT IS RELEVANT---")
            filtered_docs.append(d)

    if not filtered_docs:
        print("---GRADE: NO RELEVANT DOCUMENTS FOUND, WEB SEARCH WILL BE USED---")
        web_search_needed = True
    
    return {"documents": filtered_docs, "web_search_needed": web_search_needed}

def generate(state):
    print("---GENERATING ANSWER---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = PromptTemplate(
        template="""You are an expert pet care assistant. Use the following retrieved context to answer the user's question.
        If you don't know the answer from the context, say that you cannot find specific information in your knowledge base.
        Be concise and helpful.

        Question: {question}\nContext: {context}\nAnswer:""",
        input_variables=["question", "context"],
    )
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}

def transform_query(state):
    print("---TRANSFORMING QUERY FOR WEB SEARCH---")
    question = state["question"]
    prompt = PromptTemplate(
        template="""You are generating search queries for a web search tool. I need to find information about the following user question.
        Convert it to 1-3 effective search queries. Return a single string where queries are separated by ' OR '.

        Original question: {question}""",
        input_variables=["question"],
    )
    query_generation_chain = prompt | llm | StrOutputParser()
    better_query = query_generation_chain.invoke({"question": question})
    return {"question": better_query}

def web_search(state):
    print("---PERFORMING WEB SEARCH---")
    question = state["question"]
    search_results = web_search_tool.invoke({"query": question})
    web_content = "\n".join([d["content"] for d in search_results])
    documents = [Document(page_content=web_content)]
    return {"documents": documents}