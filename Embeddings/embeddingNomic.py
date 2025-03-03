import faiss
import getpass
import os
import pandas as pd
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


if not os.getenv("NOMIC_API_KEY"):
    os.environ["NOMIC_API_KEY"] = getpass.getpass("Enter your Nomic API key: ")


from langchain_nomic import NomicEmbeddings

embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

project_data = pd.read_csv('data/csvs/data_projects_2024_5.csv')

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(embedding_function=embeddings,index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},)

batch_size = 1000
texts = project_data['cfAbstr'].tolist()    
doc_ids = project_data['cfProjId'].tolist()

for i in range(0, len(texts), batch_size):  # Reach the 10M token limit after batch 40
    print(f"Processing batch {i // batch_size}")
    vector_store.add_texts(texts=texts[i:i+batch_size], metadatas=[{"doc_id": doc_id} for doc_id in doc_ids[i:i+batch_size]])

vector_store.save_local("faiss_index")