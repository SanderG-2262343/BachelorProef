import pandas as pd
from top2vec import Top2Vec
from langchain_chroma import Chroma

df = pd.read_csv('data_projects_2024_5.csv')

# Load your text data
print(df.count())
df = df[df['cfAbstr'].str.len() >= 20]
print(df.count())
texts = df['cfAbstr'].tolist()




# Train Top2Vec (can take time for large datasets)
top2vec_model = Top2Vec(texts, embedding_model="doc2vec")

# Get document embeddings
doc_vectors = top2vec_model.document_vectors
vector_store = Chroma(persist_directory="data_projects_2024_5_vector_store_top2vec")
vector_store._collection.upsert(texts=texts, ids=df['cfProjId'].tolist(), embeddings=doc_vectors)
