import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

df = pd.read_csv('data_publications_2024_5.csv')
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(embedding_function=embeddings,persist_directory = "data_projects_2024_5_vector_store")


df.dropna(how='any', inplace=True)


successfulmatch = 0
texts = df['cfAbstr'].tolist()
titles = df['cfTitle'].tolist()
projIds = df['cfProjId'].tolist()
for i in range(0, len(texts)):
    if i % 100 == 0:
        print(f"Processing publication {i}")
    #results = vector_store.similarity_search(titles[i] + texts[i], 5)
    results = vector_store.search(titles[i] + texts[i],'mmr',k = 5)
    for result in results:
        if result.id in projIds[i]:
            successfulmatch += 1
            break

print(f"Success Rate of: {successfulmatch * 100 / len(texts)}%")