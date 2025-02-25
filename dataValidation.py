import pandas as pd
import top2vec
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
import top2vec.top2vec

df = pd.read_csv('data_publications_2024_5_2.csv')
embeddings = OllamaEmbeddings(model="nomic-embed-text")



df.dropna(how='any', inplace=True)

#allow_dangerous_deserialization=True since we made in EmbeddingNomicLocal.py
faiss_store = FAISS.load_local("data_projects_2024_5_vector_store_TitleAbstract_faiss",embeddings=embeddings,allow_dangerous_deserialization=True)


def testEmbeddingNomic(texts,titles,projIds,vector_store):
    successfulmatch = 0
    for i in range(0, len(texts) - 10000):
        if i % 100 == 0:
            print(f"Processing publication {i}")

        
        results = vector_store.similarity_search(titles[i] + texts[i], 2)
        #results = vector_store.search(titles[i] + texts[i],'mmr',k = 5)
        for result in results:
            id = result.metadata['doc_id']
            if id in projIds[i]:
                successfulmatch += 1
                break
    return successfulmatch
    


def testTop2VecModel(texts,titles,projIds):
    vector_store = Chroma(persist_directory="data_projects_2024_5_vector_store_top2vec")
    successfulmatch = 0
    model = top2vec.top2vec.load("top2vec_model")
    for i in range(0, len(texts)):
        if i % 100 == 0:
            print(f"Processing publication {i}")
        embedding = model.model.infer_vector((titles[i] + " " + texts[i]).split())
        results = vector_store.similarity_search_by_vector(embedding, 100)
        #results = vector_store.search(titles[i] + texts[i],'mmr',k = 5)
        for result in results:
            if result.id in projIds[i]:
                successfulmatch += 1
                break
    return successfulmatch
    
texts = df['cfAbstr'].tolist()
titles = df['cfTitle'].tolist()
projIds = df['cfProjId'].tolist()
#successfulmatch = testEmbeddingNomic(texts,titles,projIds,Chroma(embedding_function=embeddings,persist_directory = "data_projects_2024_5_vector_store"))
successfulmatch = testEmbeddingNomic(texts,titles,projIds,faiss_store)
#successfulmatch = testEmbeddingNomic(texts,titles,projIds,Chroma(embedding_function=embeddings,persist_directory = "data_projects_2024_5_vector_store_TitleAbstract"))
#successfulmatch2 = testTop2VecModel(texts,titles,projIds)
print(f"Success Rate of Nomic: {successfulmatch * 100 / len(texts)}%")
#print(f"Success Rate of Top2Vec: {successfulmatch2 * 100 / len(texts)}%")
