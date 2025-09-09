from langchain_community.document_loaders import UnstructuredFileLoader

# text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# vector db
from langchain.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI

# LLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_document_chain
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st
import time

from dotenv import load_dotenv

load_dotenv()


# streamlit app title
st.title("RAG with Langchain")

urls = [
    "https://raw.githubusercontent.com/hwchase17/langchain/master/README.md",
    "https://raw.githubusercontent.com/hwchase17/langchain/master/docs/index.md", 
    
]

loader = [UnstructuredFileLoader(url) for url in urls]

data = loader.load()

print(data)


# perform chunking operation

# 1 split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=500
    )

docs = text_splitter.split_documents(data)

print("total Number of document",len(docs))


# vectorization and storing in vector db

# this is the knowledge base

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    collection_name="langchain-docs"
)

#RAG system

# R = retriever

retriever = vectorstore.as_retriever(
    search_type='similarity' ,
    search_kwargs={"k":3} # Give top 3 similar/rlevant answers
)

# test the retriever

retrieved_docs = retriever.invoke(
    "What kind of services do you provide"
)

print("Total retrieved docs",len(retrieved_docs))

# this will print all the documents so we dont require that hence we connect to an llm


# llm
llm = OpenAI(temperature=0.3, max_tokens=500)

query = st.chat_input("Ask me anything")

prompt  = query

system_prompt = """You are a helpful AI assistant that helps people find information.
Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always answer in a very concise manner.

\n\n

{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])  


if query:
    question_answer_chain = create_stuff_document_chain(llm=llm, prompt=prompt) 

    chain = create_retrieval_chain(
        llm=llm,
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
        return_source_documents=True
    )

    result = chain.invoke({
        "input": query
    })
    
    st.write("Answer: " + result['output'])

    print("Answer: ", result['output'])