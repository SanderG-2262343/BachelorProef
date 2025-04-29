import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from google import genai
from google.genai import types
import voyageai
import torch
#from langchain_community.vectorstores import FAISS
from Embeddings.embeddingVoyageAI import cleanParticipants, cleanDisciplines
import top2vec.top2vec
import os
import shutil
from dotenv import load_dotenv
import numpy as np
import voyageai.object
import time

load_dotenv()



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

def runTestsLingMinstral(publications,vector_store_directory,embeddingsSave,zipfunction  = None):
    embeddingsLingMistral = HuggingFaceEmbeddings(model_name="Linq-AI-Research/Linq-Embed-Mistral",
        model_kwargs={
        "device": "cuda",
        "model_kwargs" : {"torch_dtype": torch.float16},  # run in fp16 to save ~50% memory
    }
    )
    vector_store = Chroma(embedding_function=embeddingsLingMistral,persist_directory = vector_store_directory)
    #participants = df['participants'].tolist()
    
    for i in [1,2,3,5] + list(range(10, 20, 10)):
        if zipfunction == None:
            successfulmatch = testEmbeddingLingMinstral(publications,vector_store,i, embeddingsSave=embeddingsSave)
        else:
            successfulmatch = testEmbeddingLingMinstral(publications,vector_store,i, embeddingsSave=embeddingsSave,zipfunction=zipfunction)
        print(f"Success Rate of LingMinstral with Top {i}: {successfulmatch * 100 / len(publications['abstract'])}%")


def runTestsVoyageAi(publications,vector_store_directory,embeddingsSave,zipfunction  = None):
    embeddingsVoyage = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])
    vector_store = Chroma(embedding_function=embeddingsVoyage,persist_directory = vector_store_directory)
    #participants = df['participants'].tolist()
    
    for i in [1,2,3,5] + list(range(10, 20, 10)):
        if zipfunction == None:
            successfulmatch = testEmbeddingVoyageAI(publications,vector_store,i, embeddingsSave=embeddingsSave)
        else:
            successfulmatch = testEmbeddingVoyageAI(publications,vector_store,i, embeddingsSave=embeddingsSave,zipfunction=zipfunction)
        print(f"Success Rate of VoyageAI with Top {i}: {successfulmatch * 100 / len(publications["abstract"])}%")

def runTestGemini(abstracts,titles,projIds,vector_store_directory,embeddingsSave,zipfunction  = None):
    vector_store = Chroma(persist_directory = vector_store_directory)
    for i in [1,2,3,5] + list(range(10, 110, 10)):
        if zipfunction == None:
            successfulmatch = testEmbeddingGemini(abstracts,titles,projIds,vector_store,i,embeddingsSave)
        else:
            successfulmatch = testEmbeddingGemini(abstracts,titles,projIds,vector_store,i,embeddingsSave,zipfunction)
        print(f"Success Rate of Gemini with Top {i}: {successfulmatch * 100 / len(abstracts)}%")


def runTestsNomic(abstracts,titles,projIds,vector_store_directory,embeddingsSave,zipfunction  = None):
    NomicEmbedding = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma(embedding_function=NomicEmbedding,persist_directory = vector_store_directory)
    for i in [1,2,3,5] + list(range(10, 110, 10)):
        if zipfunction == None:
            successfulmatch = testEmbeddingNomic(abstracts,titles,projIds,vector_store,i,embeddingsSave)
        else:
            successfulmatch = testEmbeddingNomic(abstracts,titles,projIds,vector_store,i,embeddingsSave,zipfunction)
        print(f"Success Rate of Nomic with Top {i}: {successfulmatch * 100 / len(abstracts)}%")


def runTestsTop2Vec(abstracts,titles,projIds,top2vecModelFilename,zipfunction  = None):
    top2vecModel = top2vec.top2vec.load(top2vecModelFilename)
    for i in [1,2,3,5] + list(range(10, 110, 10)):
        if zipfunction == None:
            successfulmatch = testTop2VecModel(abstracts,titles,projIds,top2vecModel,i)
        else:
            successfulmatch = testTop2VecModel(abstracts,titles,projIds,top2vecModel,i,zipfunction)
        successfulmatch = testTop2VecModel(abstracts,titles,projIds,top2vecModel,i)
        print(f"Success Rate of Top2Vec with Top {i}: {successfulmatch * 100 / len(abstracts)}%")


def testEmbeddingLingMinstral(publications,vector_store,top_k = 2,embeddingsSave = "placeholder" ,
                          zipfunction = lambda titles, abstracts,participants,disciplines,dataProviders: ["Instruct: Compare this publication with a project \n Query:" + title + " " + abstract for title, abstract in zip(titles, abstracts)]):
    successfulmatch = 0
    embeddingsLingMistral = HuggingFaceEmbeddings(model_name="Linq-AI-Research/Linq-Embed-Mistral",
        model_kwargs={
        "device": "cuda",
        "model_kwargs" : {"torch_dtype": torch.float16},  # run in fp16 to save ~50% memory
    }
    )

    #voyageaiEmbed = voyageai.Client(api_key=os.environ['VOYAGE_API_KEY'])

    #store embeddings locally for multiple runs
    titles = publications['title'].tolist()
    abstracts = publications['abstract'].tolist()
    projIds = publications['projId'].tolist()
    participants = publications['participants'].tolist()
    participants = [cleanParticipants(participant) for participant in participants]
    disciplines = publications['flemishDisciplines'].tolist()
    disciplines = [cleanDisciplines(discipline) for discipline in disciplines] #remove prefix code
    dataProviders = publications['dataProvider'].tolist()
    combined = zipfunction(titles, abstracts, participants, disciplines, dataProviders)

    

    if not os.path.exists(embeddingsSave):
        embeddingsLingMistral.batch_size = 128
        embeddings = embeddingsLingMistral.embed_documents(combined)
        pd.DataFrame(embeddings).to_csv(embeddingsSave,index=False)
    else:
        embeddings = pd.read_csv(embeddingsSave).values.tolist()

    for i in range(0, len(abstracts)):
        results = vector_store.similarity_search_by_vector(embeddings[i], top_k)
        for result in results:
            #id = result.metadata['doc_id']
            if result.id in projIds[i]:
                successfulmatch += 1
                break
    return successfulmatch


def testEmbeddingVoyageAI(publications,vector_store,top_k = 2,embeddingsSave = "placeholder" ,
                          zipfunction = lambda titles, abstracts,participants,disciplines,dataProviders: ["Instruct: Compare this publication with a project \n Query:" + title + " " + abstract for title, abstract in zip(titles, abstracts)]):
    successfulmatch = 0
    embeddingsVoyage = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])

    #voyageaiEmbed = voyageai.Client(api_key=os.environ['VOYAGE_API_KEY'])

    #store embeddings locally for multiple runs
    titles = publications['title'].tolist()
    abstracts = publications['abstract'].tolist()
    projIds = publications['projId'].tolist()
    participants = publications['participants'].tolist()
    participants = [cleanParticipants(participant) for participant in participants]
    disciplines = publications['flemishDisciplines'].tolist()
    disciplines = [cleanDisciplines(discipline) for discipline in disciplines] #remove prefix code
    dataProviders = publications['dataProvider'].tolist()
    combined = zipfunction(titles, abstracts, participants, disciplines, dataProviders)
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
    #reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    successfulmatch2 = 0
    for i in range(0, len(abstracts)):
        #if i % 100 == 0:
            #print(f"Processing publication {i}")

        results = vector_store.similarity_search_by_vector(embeddings[i], top_k,
                                                           filter = {"dataProvider": dataProviders[i]}
                                                           )
        #results = vector_store.max_marginal_relevance_search_by_vector(embeddings[i],top_k)
        #results = vector_store.similarity_search(titles[i] + texts[i], 2)
        #results = vector_store.search(titles[i] + texts[i],'mmr',k = 5)

        #Code for reranker
        '''
        documents = []
        for result in results:
            documents.append(result.page_content)
            
           
            
        reranking =  reranker.rank(combined[i],documents,top_k=3,return_documents=True)
        for r in reranking:
            if results[r["corpus_id"]].id in projIds[i]:
                successfulmatch2 += 1
                break
        time.sleep(1)


        ''' 
        oldsuccessmatch = successfulmatch
        for result in results:
            #id = result.metadata['doc_id']
            if result.id in projIds[i]:
                successfulmatch += 1
                break
        #if top_k == 20 and successfulmatch == oldsuccessmatch:
        #    print(projIds[i],titles[i],",".join([result.id for result in results]))
        
    
    #return successfulmatch2
    return successfulmatch


def testEmbeddingGemini(abstracts,titles,projIds,vector_store,top_k = 2,embeddingsSave = "placeholder", zipfunction = lambda titles, abstracts: [title + " " + abstract for title, abstract in zip(titles, abstracts)]):
    successfulmatch = 0
    embeddingsGemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    combined = zipfunction(titles, abstracts)
    if not os.path.exists(embeddingsSave):
        embeddings = []
        for i in range(0,len(abstracts),5):
            print(f"Processing batch {i}")
            result = embeddingsGemini.models.embed_content(model="gemini-embedding-exp-03-07",contents=combined[i:i+5])
            embeddings += [embedding.values for embedding in result.embeddings]
            time.sleep(60)
        pd.DataFrame(embeddings).to_csv(embeddingsSave,index=False)
    else:
        embeddings = pd.read_csv(embeddingsSave).values.tolist()
    for i in range(0, len(abstracts)):
        results = vector_store.similarity_search_by_vector(embeddings[i], top_k)

        oldsuccessmatch = successfulmatch
        for result in results:
            #id = result.metadata['doc_id']
            if result.id in projIds[i]:
                successfulmatch += 1
                break
        if top_k == 30 and successfulmatch == oldsuccessmatch:
            print(projIds[i],titles[i])
    return successfulmatch

    
def testEmbeddingNomic(abstracts,titles,projIds,vector_store,top_k = 2,embeddingsSave = "placeholder", zipfunction = lambda titles, abstracts: [title + " " + abstract for title, abstract in zip(titles, abstracts)]):
    successfulmatch = 0
    NomicEmbedding = OllamaEmbeddings(model="nomic-embed-text")

    #store embeddings locally for multiple runs
    combined = zipfunction(titles, abstracts)
    if not os.path.exists(embeddingsSave):
        embeddings = NomicEmbedding.embed_documents(combined)
        pd.DataFrame(embeddings).to_csv(embeddingsSave,index=False)
        
    else:
        embeddings = pd.read_csv(embeddingsSave).values.tolist()


    for i in range(0, len(abstracts)):
        #if i % 100 == 0:
            #print(f"Processing publication {i}")

        results = vector_store.similarity_search_by_vector(embeddings[i], top_k)
        #results = vector_store.similarity_search(titles[i] + " " + texts[i], 3)
        
        #results = vector_store.similarity_search(titles[i] + texts[i], 2)
        #results = vector_store.search(titles[i] + texts[i],'mmr',k = 5)
        oldsuccessmatch = successfulmatch
        for result in results:
            #id = result.metadata['doc_id']
            if result.id in projIds[i]:
                successfulmatch += 1
                break
        #if top_k == 100 and successfulmatch == oldsuccessmatch:
        #    print(projIds[i])
    return successfulmatch
    


def testTop2VecModel(texts,titles,projIds,top2vecModel,top_k = 2,zipfunction = lambda titles, abstracts: [title + " " + abstract for title, abstract in zip(titles, abstracts)]):
    successfulmatch = 0
    combined = zipfunction(titles, texts)
    for i in range(0, len(combined)):
        #if i % 100 == 0:
            #print(f"Processing publication {i}")
        
        results = top2vecModel.query_documents(combined[i],num_docs=top_k)

        #results = vector_store.similarity_search_by_vector(embedding, 100)
        #results = vector_store.search(titles[i] + texts[i],'mmr',k = 5)
        oldsuccessmatch = successfulmatch
        for id in results[2]:
            if id in projIds[i]:
                successfulmatch += 1
                break
        #if top_k == 100 and successfulmatch == oldsuccessmatch:
        #    print(projIds[i])
    return successfulmatch
    

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
                top2vecModelFilename, publicationDataFileLocation, zipfunctions = None):
    df = pd.read_csv(publicationDataFileLocation)
    abstracts = df['abstract'].tolist()
    titles = df['title'].tolist()
    projIds = df['projId'].tolist()

    if zipfunctions is not None and len(zipfunctions) == 3:
        runTestsVoyageAi(df, vector_store_directoryVoyage, embeddingSaveDirectoryVoyage, zipfunction=zipfunctions[0])
        runTestsNomic(abstracts, titles, projIds, vector_store_directoryNomic, embeddingsSaveDirectoryNomic, zipfunction=zipfunctions[1])
        runTestsTop2Vec(abstracts, titles, projIds, top2vecModelFilename,zipfunction=zipfunctions[2])
    else:
        runTestsVoyageAi(df, vector_store_directoryVoyage, embeddingSaveDirectoryVoyage)
        runTestsNomic(abstracts, titles, projIds, vector_store_directoryNomic, embeddingsSaveDirectoryNomic)
        runTestsTop2Vec(abstracts, titles, projIds, top2vecModelFilename)

    
    

#mergeEmbeddingsVectorStore()
#runTestsVoyageAi(abstracts,titles,projIds, "data/vectorStores/data_projects_2024_5_vector_store_VoyageAI","data/embeddingSaves/embeddingsVoyage.csv")
#runTestsNomic(texts,titles,projIds, "data/vectorStores/data_projects_2024_5_vector_store_TitleAbstract","data/embeddingSaves/embeddingsNomic.csv")
#runTestsTop2Vec(texts,titles,projIds)


#df = pd.read_csv("data/csvs/data_publications_2024_5_TestSample.csv")
#abstracts = df['abstract'].tolist()
#titles = df['title'].tolist()
#projIds = df['projId'].tolist()
#runTestGemini(abstracts,titles,projIds,"data/vectorStores/data_projects_2024_5_vector_store_Gemini_exp_TestSample","data/embeddingSaves/embeddingsGemini_exp.csv")

#print(time.ctime())
#sucessrate = testEmbeddingVoyageAI(abstracts,titles,projIds,Chroma(embedding_function=VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY']),persist_directory = "data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TestSample"),3,embeddingsSave="data/embeddingSaves/embeddingsVoyage_TestSample.csv")
#print(time.ctime())
#print(f"Success Rate of VoyageAI with Top 30: {sucessrate * 100 / len(abstracts)}%")