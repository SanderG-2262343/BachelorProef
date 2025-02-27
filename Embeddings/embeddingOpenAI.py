#key not currently available

import pandas as pd
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from dotenv import load_dotenv
import getpass
import os
load_dotenv()


if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

project_data = pd.read_csv('data_projects_2024_5.csv')

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(embedding_function=embeddings,index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},)

batch_size = 1000
texts = project_data['cfAbstr'].tolist()    
doc_ids = project_data['cfProjId'].tolist()

for i in range(0, len(texts), batch_size):
    print(f"Processing batch {i // batch_size}")
    vector_store.add_texts(texts=texts[i:i+batch_size], metadatas=[{"doc_id": doc_id} for doc_id in doc_ids[i:i+batch_size]])

vector_store.save_local("faiss_index")

