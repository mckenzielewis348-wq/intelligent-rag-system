import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# 1. Load document and break into chunks
loader = TextLoader("knowledge.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# 2. Convert text to embeddings and store in ChromaDB vector database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Create prompt template for context-grounded answers
prompt = ChatPromptTemplate.from_template(
    "Answer the question based ONLY on the context below:\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 4. Construct RAG execution chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Execute query
if __name__ == "__main__":
    query = "How are alerts delivered when an issue is detected?"
    response = rag_chain.invoke(query)
    print("\nResult:\n", response)
