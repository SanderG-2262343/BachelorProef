import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

df = pd.read_csv('data_publications_2024_5.csv')
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(embedding_function=embeddings,persist_directory = "data_projects_2024_5_vector_store")


df.dropna(how='any', inplace=True)

print(df.columns)
successfulmatch = 0
texts = df['cfAbstr'].tolist()
titles = df['cfTitle'].tolist()
projIds = df['cfProjId'].tolist()
for i in range(0, 1):
    if i % 100 == 0:
        print(f"Processing publication {i}")
    results = vector_store.similarity_search(titles[i] + texts[i], 5)
    print(results)
    for result in results:
        print(result.id, projIds[i])
        if result.id in projIds[i]:
            successfulmatch += 1
            break

print(f"Success Rate of: {successfulmatch % len(df)}")