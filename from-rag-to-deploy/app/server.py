# server.py

from typing import List, Union


from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI()

load_dotenv()

# Ruta al archivo PDF
file_path = "./data/Bilingual_disadvantages_are_systematically_compensated_by_bilingual_advantages_across_tasks_and_populations.pdf"

# Cargar el archivo PDF
loader = PyPDFLoader(file_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Construímos el retriever
retriever = vectorstore.as_retriever()
retriever.vectorstore.add_documents(docs)

# Prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/playground")

# add_routes(app, NotImplemented)
add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)