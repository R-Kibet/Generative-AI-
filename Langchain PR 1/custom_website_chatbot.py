# -*- coding: utf-8 -*-
"""

Custom Website chatbot.ipynb

"""

!pip -q install langchain langchain-community
!pip -q install pypdf
!pip -q install sentence_transformers
!pip  install openai
!pip  install tiktoken

!pip install tokenizers
!pip install faiss-cpu
!pip -q install unstructured

!pip install numpy
!pip install nltk

import sys
import os
import torch
import textwrap
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

os.environ['OPENAI_API_KEY'] = ""

"""## 1 Pass the url links

this will enable getting all necessary information
"""

# can support multiple urls
URLs = [
    "https://any url"
]

"""### 2 Now we extract the document data"""

loaders = UnstructuredURLLoader(urls=URLs)
data = loaders.load()

data

len(data)

"""## CHUNKING"""

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200)

chunks = text_splitter.split_documents(data)

len(chunks)

chunks[0]

"""## 3 Initialize embedding model"""

embeddings = OpenAIEmbeddings()

query_result = embeddings.embed_query("Hello world")

"""## 4 VectorDB saving to vector store"""

vectorstore = FAISS.from_documents(chunks, embeddings)

"""## 5 initialize large language model"""

llm = ChatOpenAI()

# testing if the llm is working fine
llm.predict("Give me player results of the nba finals game 6")

"""# We combine both llm and the vectordb to retrieve information and give answer accordingly this is RAG"""

chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

"""## 6 we create a prompt"""

while True:
  query = input(f"Prompt: ")
  if query == 'exit':
    print('Exiting')
    sys.exit()
  if query == '':
    continue
  result = chain({"question": query}, return_only_outputs=True)
  print(textwrap.fill(result['answer'], 100))
  print(textwrap.fill(result['sources'], 100))
