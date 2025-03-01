import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
#from langchain_community.vectorstores import FAISS
import top2vec.top2vec
import os
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv('data_publications_2024_5.csv')
NomicEmbedding = OllamaEmbeddings(model="nomic-embed-text")




df.dropna(how='any', inplace=True)

#allow_dangerous_deserialization=True since we made in EmbeddingNomicLocal.py
#faiss_store = FAISS.load_local("data_projects_2024_5_vector_store_TitleAbstract_faiss",embeddings=embeddings,allow_dangerous_deserialization=True)

def runTestsVoyageAi(texts,titles,projIds):
    embeddingsVoyage = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])
    vector_store = Chroma(embedding_function=embeddingsVoyage,persist_directory = "data_projects_2024_5_vector_store_VoyageAI")
    for i in [1,2,3,5]:
        successfulmatch = testEmbeddingVoyageAI(texts,titles,projIds,vector_store,i)
        print(f"Success Rate of VoyageAI with Top {i}: {successfulmatch * 100 / len(texts)}%")

    for i in range(10, 110, 10):
        successfulmatch = testEmbeddingVoyageAI(texts,titles,projIds,vector_store,i)
        print(f"Success Rate of VoyageAI with Top {i}: {successfulmatch * 100 / len(texts)}%")

def runTestsNomic(texts,titles,projIds):
    vector_store = Chroma(embedding_function=NomicEmbedding,persist_directory = "data_projects_2024_5_vector_store_TitleAbstract")
    for i in [1,2,3,5]:
        successfulmatch = testEmbeddingNomic(texts,titles,projIds,vector_store,i)
        print(f"Success Rate of Nomic with Top {i}: {successfulmatch * 100 / len(texts)}%")

    for i in range(10, 110, 10):
        successfulmatch = testEmbeddingNomic(texts,titles,projIds,vector_store,i)
        print(f"Success Rate of Nomic with Top {i}: {successfulmatch * 100 / len(texts)}%")

def runTestsTop2Vec(texts,titles,projIds):
    for i in [1,2,3,5]:
        successfulmatch = testTop2VecModel(texts,titles,projIds,i)
        print(f"Success Rate of Top2Vec with Top {i}: {successfulmatch * 100 / len(texts)}%")

    for i in range(10, 110, 10):
        successfulmatch = testTop2VecModel(texts,titles,projIds,i)
        print(f"Success Rate of Top2Vec with Top {i}: {successfulmatch * 100 / len(texts)}%")

def testEmbeddingVoyageAI(texts,titles,projIds,vector_store,top_k = 2):
    successfulmatch = 0
    embeddingsVoyage = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])

    #store embeddings locally for multiple runs
    #combined = ["Instruct: Compare this publication with a project \n Query:" + title + " " + text for title, text in zip(titles, texts)]
    combined = [title + " " + text for title, text in zip(titles, texts)]

    if not os.path.exists("embeddingsVoyageI.csv"):
        embeddingsVoyage.batch_size = 128
        embeddings = embeddingsVoyage.embed_documents(combined)
        pd.DataFrame(embeddings).to_csv("embeddingsVoyageI.csv",index=False)
    else:
        embeddings = pd.read_csv("embeddingsVoyageI.csv").values.tolist()


    
    #reranker = voyageai.Client()
    for i in range(0, len(texts)):
        #if i % 100 == 0:
            #print(f"Processing publication {i}")

        results = vector_store.similarity_search_by_vector(embeddings[i], top_k)
        #results = vector_store.similarity_search(titles[i] + texts[i], 2)
        #results = vector_store.search(titles[i] + texts[i],'mmr',k = 5)

        #Code for reranker
        """
        documents = []
        for result in results:
            documents.append(result.page_content)
        
        reranking = reranker.rerank(combined[i],documents,model="rerank-2",top_k=5)

        for r in reranking.results:
            if results[r.index].id in projIds[i]:
                successfulmatch += 1
                break
        time.sleep(1)
        """        

        for result in results:
            #id = result.metadata['doc_id']
            if result.id in projIds[i]:
                successfulmatch += 1
                break
    return successfulmatch

def testEmbeddingNomic(texts,titles,projIds,vector_store,top_k = 2):
    successfulmatch = 0

    #store embeddings locally for multiple runs
    combined = [title + " " + text for title, text in zip(titles, texts)]
    if not os.path.exists("embeddingsNomic.csv"):
        embeddings = NomicEmbedding.embed_documents(combined)
        pd.DataFrame(embeddings).to_csv("embeddingsNomic.csv",index=False)
    else:
        embeddings = pd.read_csv("embeddingsNomic.csv").values.tolist()


    for i in range(0, len(texts)):
        #if i % 100 == 0:
            #print(f"Processing publication {i}")

        results = vector_store.similarity_search_by_vector(embeddings[i], top_k)
        #results = vector_store.similarity_search(titles[i] + " " + texts[i], 3)
        
        #results = vector_store.similarity_search(titles[i] + texts[i], 2)
        #results = vector_store.search(titles[i] + texts[i],'mmr',k = 5)
        for result in results:
            #id = result.metadata['doc_id']
            if result.id in projIds[i]:
                successfulmatch += 1
                break
    return successfulmatch
    


def testTop2VecModel(texts,titles,projIds,top_k = 2):
    successfulmatch = 0
    model = top2vec.top2vec.load("top2vec_model_deep-learn")
    for i in range(0, len(texts)):
        #if i % 100 == 0:
            #print(f"Processing publication {i}")
        results = model.query_documents((titles[i] + " " + texts[i]),num_docs=top_k)

        #results = vector_store.similarity_search_by_vector(embedding, 100)
        #results = vector_store.search(titles[i] + texts[i],'mmr',k = 5)
        for id in results[2]:
            if id in projIds[i]:
                successfulmatch += 1
                break
    return successfulmatch
    
texts = df['cfAbstr'].tolist()
titles = df['cfTitle'].tolist()
projIds = df['cfProjId'].tolist()
#successfulmatch = testEmbeddingNomic(texts,titles,projIds,Chroma(embedding_function=embeddings,persist_directory = "data_projects_2024_5_vector_store"))
#successfulmatch = testEmbeddingNomic(texts,titles,projIds,faiss_store)
#successfulmatch = testEmbeddingNomic(texts,titles,projIds,Chroma(embedding_function=NomicEmbedding,persist_directory = "data_projects_2024_5_vector_store"),2)
#embeddingsVoyage = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])
#successfulmatch = testEmbeddingVoyageAI(texts,titles,projIds,Chroma(embedding_function=embeddingsVoyage,persist_directory = "data_projects_2024_5_vector_store_VoyageAI"),100)
#successfulmatch2 = testTop2VecModel(texts,titles,projIds)
#print(f"Success Rate of Nomic: {successfulmatch * 100 / len(texts)}%")
#print(f"Success Rate of Top2Vec: {successfulmatch2 * 100 / len(texts)}%")

#runTestsVoyageAi(texts,titles,projIds)
#runTestsNomic(texts,titles,projIds)
runTestsTop2Vec(texts,titles,projIds)

