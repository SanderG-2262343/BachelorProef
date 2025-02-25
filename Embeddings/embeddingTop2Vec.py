import pandas as pd
import os
from top2vec import Top2Vec
from langchain_chroma import Chroma
from langchain_core.documents import Document


df = pd.read_csv('data_projects_2024_5.csv')
df.dropna(how='any', inplace=True)

# Load your text data
print(df.count())
df = df[df['cfAbstr'].str.len() >= 20]
print(df.count())
texts = df['cfAbstr'].tolist()
titles = df['cfTitle'].tolist()
combined = [title + " " + text for title, text in zip(titles, texts)]




# Train Top2Vec (can take time for large datasets)
if os.path.exists("top2vec_model"):
    top2vec_model = Top2Vec.load("top2vec_model")
else:
    top2vec_model = Top2Vec(combined, embedding_model="doc2vec")

# Get document embeddings
doc_vectors = top2vec_model.document_vectors
vector_store = Chroma(persist_directory="data_projects_2024_5_vector_store_top2vec")
for i in range(0, len(doc_vectors), 1000):
    print(f"Inserting embeddings {i} to {i+1000}")
    documents = []

    vector_store._collection.upsert(ids=df['cfProjId'].tolist()[i:i+1000], embeddings=doc_vectors[i:i+1000],documents=texts[i:i+1000])
top2vec_model.save("top2vec_model")
