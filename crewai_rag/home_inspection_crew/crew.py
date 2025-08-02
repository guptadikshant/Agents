from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List

# Set your API keys
os.environ["GROQ_API_KEY"] = "your-groq-api-key"

# Initialize LLM
llm = ChatGroq(
    model_name="llama2-70b-4096",
    temperature=0.3,
    max_tokens=4096
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Qdrant
qdrant = Qdrant.from_documents(
    [],  # Empty initial documents
    embeddings,
    location=":memory:",  # Use disk-based storage with "local" instead of ":memory:"
    collection_name="my_documents"
)

class RAGCrew:
    def __init__(self):
        # Create agents
        self.researcher = Agent(
            role='Research Analyst',
            goal='Analyze and store documents in the vector database',
            backstory='Expert at processing and analyzing documents for RAG systems',
            allow_delegation=True,
            llm=llm
        )
        
        self.answering_agent = Agent(
            role='Question Answering Expert',
            goal='Provide accurate answers based on stored knowledge',
            backstory='Specialist in retrieving and synthesizing information from the vector database',
            allow_delegation=True,
            llm=llm
        )

    def store_documents(self, documents: List[str]):
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.create_documents(documents)
        
        # Store in Qdrant
        qdrant.add_documents(splits)
        
        task = Task(
            description="Process and verify document storage",
            agent=self.researcher
        )
        return task.execute()

    def answer_question(self, question: str):
        # Search relevant documents
        docs = qdrant.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        task = Task(
            description=f"Answer the following question using the context provided: {question}\nContext: {context}",
            agent=self.answering_agent
        )
        return task.execute()

# Usage example
if __name__ == "__main__":
    rag_crew = RAGCrew()
    
    # Store documents
    documents = [
        "This is a sample document about AI.",
        "Another document about machine learning."
    ]
    rag_crew.store_documents(documents)
    
    # Ask questions
    answer = rag_crew.answer_question("What is AI?")
    print(f"Answer: {answer}")