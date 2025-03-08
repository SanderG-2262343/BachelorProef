import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
import voyageai
#from langchain_community.vectorstores import FAISS
import top2vec.top2vec
import os
import shutil
from dotenv import load_dotenv
import numpy as np
import voyageai.object

load_dotenv()

df = pd.read_csv('data/csvs/data_publications_2024_5_FRIS_matched.csv')
NomicEmbedding = OllamaEmbeddings(model="nomic-embed-text")




df.dropna(how='any', inplace=True,subset=['cfProjId'])


# Merge the title embedding and abstract embedding into one combined embedding using (Concatenation, Sum, Average, etc.) depending on what currently testing
# Then store the combined embedding in a vector store (data_projects_2024_5_vector_store_combined) for future use
def mergeEmbeddingsVectorStore():
    embeddingsVoyage = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])
    vector_store1=Chroma(embedding_function=embeddingsVoyage,persist_directory = "data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_Title")
    vector_store2=Chroma(embedding_function=embeddingsVoyage,persist_directory = "data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_Abstract")

    if os.path.exists("data/vectorStores/data_projects_2024_5_vector_store_combined"):
        shutil.rmtree("data/vectorStores/data_projects_2024_5_vector_store_combined")
    vector_store_combined = Chroma(embedding_function=embeddingsVoyage,persist_directory = "data/vectorStores/data_projects_2024_5_vector_store_combined")
    data1 = vector_store1.get(include = ["embeddings"])
    for i in range(0, len(data1['embeddings']),1000):  #Cant get all at once

        data2 = (vector_store2.get(include = ["embeddings"],ids = data1['ids'][i:i+1000]))
        embeddings = [np.add(0.2 *a, 0.8 * b) for a, b in zip(data1['embeddings'][i:i+1000], data2['embeddings'])]
        print(len(embeddings[0]))
        #vector_store_combined.add(embeddings,ids = data1['ids'])

        embeddings = embeddings / np.linalg.norm(embeddings,axis=1, keepdims=True)
        vector_store_combined._collection.add(ids=data1['ids'][i:i+1000], embeddings=embeddings,documents=[""] * len(data1['embeddings'][i:i+1000]))  #add without document
    
#allow_dangerous_deserialization=True since we made in EmbeddingNomicLocal.py
#faiss_store = FAISS.load_local("data_projects_2024_5_vector_store_TitleAbstract_faiss",embeddings=embeddings,allow_dangerous_deserialization=True)


def runTestsVoyageAi(abstracts,titles,projIds,vector_store_directory,embeddingsSave,zipfunction  = None):
    embeddingsVoyage = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])
    vector_store = Chroma(embedding_function=embeddingsVoyage,persist_directory = vector_store_directory)
    #participants = df['participants'].tolist()
    

    for i in [1,2,3,5] + list(range(10, 110, 10)):
        if zipfunction == None:
            successfulmatch = testEmbeddingVoyageAI(abstracts,titles,projIds,vector_store,i, embeddingsSave=embeddingsSave)
        else:
            successfulmatch = testEmbeddingVoyageAI(abstracts,titles,projIds,vector_store,i, embeddingsSave=embeddingsSave,zipfunction=zipfunction)
        print(f"Success Rate of VoyageAI with Top {i}: {successfulmatch * 100 / len(abstracts)}%")

def runTestsNomic(abstracts,titles,projIds,vector_store_directory,embeddingsSave,zipfunction  = None):
    vector_store = Chroma(embedding_function=NomicEmbedding,persist_directory = vector_store_directory)
    for i in [1,2,3,5] + list(range(10, 110, 10)):
        if zipfunction == None:
            successfulmatch = testEmbeddingNomic(abstracts,titles,projIds,vector_store,i,embeddingsSave)
        else:
            successfulmatch = testEmbeddingNomic(abstracts,titles,projIds,vector_store,i,embeddingsSave,zipfunction)
        print(f"Success Rate of Nomic with Top {i}: {successfulmatch * 100 / len(abstracts)}%")

def runTestsTop2Vec(abstracts,titles,projIds,top2vecModelFilename):
    top2vecModel = top2vec.top2vec.load(top2vecModelFilename)
    for i in [1,2,3,5] + list(range(10, 110, 10)):
        successfulmatch = testTop2VecModel(abstracts,titles,projIds,top2vecModel,i)
        print(f"Success Rate of Top2Vec with Top {i}: {successfulmatch * 100 / len(abstracts)}%")

def testEmbeddingVoyageAI(abstracts,titles,projIds,vector_store,top_k = 2,embeddingsSave = "placeholder" ,zipfunction = lambda titles, abstracts: ["Instruct: Compare this publication with a project \n Query:" + title + " " + abstract for title, abstract in zip(titles, abstracts)]):
    successfulmatch = 0
    embeddingsVoyage = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])

    voyageaiEmbed = voyageai.Client(api_key=os.environ['VOYAGE_API_KEY'])

    #store embeddings locally for multiple runs
    combined = zipfunction(titles, abstracts)
    #combined = [title + " " + text for title, text in zip(titles, texts)]

    

    if not os.path.exists(embeddingsSave):
        embeddingsVoyage.batch_size = 128
        #combined = [f"Instruct: Compare this publication with a project \n Query: Title: {title} Participants: {participants} Abstract: {text}" for title, text,participants in zip(titles, texts,participants)]
        #embeddings = embeddingsVoyage.embed_query(combined)
        embeddings = embeddingsVoyage.embed_documents(combined)
        pd.DataFrame(embeddings).to_csv(embeddingsSave,index=False)
    else:
        embeddings = pd.read_csv(embeddingsSave).values.tolist()

    '''
    embeddingsSave2 = "embeddingsVoyage_Abstract.csv"
    embeddingsSave2 = "data/embeddingSaves/" + embeddingsSave2

    if not os.path.exists(embeddingsSave2):
        embeddingsVoyage.batch_size = 128
        #combined = [f"Instruct: Compare this publication with a project \n Query: Title: {title} Participants: {participants} Abstract: {text}" for title, text,participants in zip(titles, texts,participants)]
        embeddings2 = embeddingsVoyage.embed_documents(texts)
        pd.DataFrame(embeddings2).to_csv(embeddingsSave2,index=False)
    else:
        embeddings2 = pd.read_csv(embeddingsSave2).values.tolist()


    embeddings = np.array(embeddings)
    embeddings2 = np.array(embeddings2)
    
    embeddingsCombined = [np.add(0.2* a, 0.8 * b) for a, b in zip(embeddings, embeddings2)]
    embeddingsCombined = embeddingsCombined / np.linalg.norm(embeddingsCombined,axis=1, keepdims=True)
    '''
    #reranker = voyageai.Client()
    for i in range(0, len(abstracts)):
        #if i % 100 == 0:
            #print(f"Processing publication {i}")

        results = vector_store.similarity_search_by_vector(embeddings[i], top_k)
        #results = vector_store.max_marginal_relevance_search_by_vector(embeddings[i],top_k)
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

def testEmbeddingNomic(texts,abstracts,projIds,vector_store,top_k = 2,embeddingsSave = "placeholder", zipfunction = lambda titles, abstracts: [title + " " + abstract for title, abstract in zip(titles, abstracts)]):
    successfulmatch = 0

    #store embeddings locally for multiple runs

    combined = zipfunction(titles, abstracts)
    if not os.path.exists(embeddingsSave):
        embeddings = NomicEmbedding.embed_documents(combined)
        pd.DataFrame(embeddings).to_csv(embeddingsSave,index=False)
        
    else:
        embeddings = pd.read_csv(embeddingsSave).values.tolist()


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
    


def testTop2VecModel(texts,titles,projIds,top2vecModel,top_k = 2,):
    successfulmatch = 0
    for i in range(0, len(texts)):
        #if i % 100 == 0:
            #print(f"Processing publication {i}")
        results = top2vecModel.query_documents((titles[i] + " " + texts[i]),num_docs=top_k)

        #results = vector_store.similarity_search_by_vector(embedding, 100)
        #results = vector_store.search(titles[i] + texts[i],'mmr',k = 5)
        for id in results[2]:
            if id in projIds[i]:
                successfulmatch += 1
                break
    return successfulmatch
    
abstracts = df['cfAbstr'].tolist()
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


def runAllTests(vector_store_directoryVoyage, embeddingSaveDirectoryVoyage, 
                vector_store_directoryNomic, embeddingsSaveDirectoryNomic, 
                top2vecModelFilename, zipfunctions = None):

    if zipfunctions is not None and len(zipfunctions) == 2:
        runTestsVoyageAi(abstracts, titles, projIds, vector_store_directoryVoyage, embeddingSaveDirectoryVoyage, zipfunctions=zipfunctions[0])
        runTestsNomic(abstracts, titles, projIds, vector_store_directoryNomic, embeddingsSaveDirectoryNomic, zipfunctions=zipfunctions[1])
    else:
        runTestsVoyageAi(abstracts, titles, projIds, vector_store_directoryVoyage, embeddingSaveDirectoryVoyage)
        runTestsNomic(abstracts, titles, projIds, vector_store_directoryNomic, embeddingsSaveDirectoryNomic)

    runTestsTop2Vec(abstracts, titles, projIds, top2vecModelFilename)

#mergeEmbeddingsVectorStore()
#runTestsVoyageAi(abstracts,titles,projIds, "data/vectorStores/data_projects_2024_5_vector_store_VoyageAI","data/embeddingSaves/embeddingsVoyage.csv")
#runTestsNomic(texts,titles,projIds, "data/vectorStores/data_projects_2024_5_vector_store_TitleAbstract","data/embeddingSaves/embeddingsNomic.csv")
#runTestsTop2Vec(texts,titles,projIds)

