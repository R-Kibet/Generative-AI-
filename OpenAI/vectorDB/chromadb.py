# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1s0o5E-XdrqZCyfCgrNi6q73AdIfzg9ro


!pip -q install chromadb openai langchain tiktoken

!pip show chromadb

!wget -q # add link or content zipped file the contnent

!unzip -q # select content to unzip

# add openai api key
"""
import os


os.environ["OPENAI_API_KEY"] = "the value of the key"

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader , TextLoader

"""## LOAD DATA"""

loader = DirectoryLoader('/content', glob = "./*.txt", loader_cls = TextLoader)

doc = loader.load()

doc

"""## Split in chunks

We split to chunks as there is a limit to input size in an LLM model creating issues

Chunks size - max amount of charecter in a chunk
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

txt_split = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

txt = txt_split.txt_split(doc)

"""## Creating DB"""

from langchain import embeddings

persist_dir = 'db'

embeddings = OpenAIEmbeddings()

# create a vector instance
vectordb = Chroma.from_documents(documents = txt,
                                 embedding = embeddings,
                                 persist_directory = persist_dir)

# persist the db to disk
vectordb.persist()
vectordb = None

"""## Loading the persistent database from disk and use it as normal"""

vectordb = Chroma(persist_directory = persist_dir,
                  embedding_function = embeddings)

"""## Make a retriever"""

retriever = vectordb.as_retriever()


docs = retriever.get_relevant_documents("write what you want to know")


retriever = vectordb.as_retriever(search_type = "similarity", search_kwargs = {"k": 2})

"""## Make a chain"""

from langchain.chains import RetrievalQA

llm = OpenAI(temperature = 0)

# create the chain to answer questions

qa_chain = RetrievalQA.from_chain_type(llm = llm,
                                       chain_type = "stuff",
                                       retriever = retriever,
                                       return_source_documents = True)

# for readability purposes
def process_response(llm_response):
    print(llm_response["result"])
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])

# example code

query = "write what you want to know"
result = qa_chain({"query": query})
result["result"]


# make readable
process_response(result)