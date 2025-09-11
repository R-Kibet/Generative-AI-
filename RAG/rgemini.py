# load data from a pdf file
from langchain_community.document_loaders import PyPDFLoader 

# chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

#   embeddings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# RAG
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st

#load api key
from dotenv import load_dotenv


load_dotenv()

st.title("RAG with Gemini")

"""
we are going to load a pdf file and from it we shall use RAG and gemini to crete a qa chatbot

"""

# 1: load the pdf
loader = PyPDFLoader("AWS.pdf")
data = loader.load()
print(f"you have {len(data)} document in your data")


# 2: chunk the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = text_splitter.split_documents(data)
print(f"Now you have {len(docs)} documents after splitting")


# 3: create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector = embeddings.embed_query("What is AWS?")
print(vector[:5])

# 4: knowledge base 
vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="chroma_db")

#5: Retrieval QA chain
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":3})

retriver_docs = retriever.invoke("What is AWS?")

print(f"we have {len(retriver_docs)} documents from the retriever")

print(retriver_docs[0].page_content)

# 6: LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.2, max_tokens=500)

# 7: RAG chain

query = st.chat_input("Ask me anything about AWS")

prompt = query  

system_prompt =  """
You are a helpful assistant that helps people find information about AWS from the provided context.
If you don't know the answer, just say that you don't know
answer in a concise manner
\n\n

{context}
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])

if query:
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    chain = create_retrieval_chain(llm=llm, retriever=retriever, return_source_documents=True, combine_documents_chain=question_answer_chain)

    response = chain.invoke({"input": query})
    print(response['answer'])
    
    st.write(response['answer'])
    with st.expander("Source Documents"):
        st.write(response['source_documents'])    