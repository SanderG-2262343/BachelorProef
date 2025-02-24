import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
import asyncio
import os

# Load the data assuming the data is already preprocessed
project_data = pd.read_csv('data_projects_2024_5.csv')


# Drop rows with empty abstracts
project_data = project_data[project_data['cfAbstr'].str.len() >= 20] 
project_data.dropna(how='any', inplace=True) #one project with no abstract


# Initialize the OllamaEmbeddings object assume the model already pulled
embeddings = OllamaEmbeddings(model="nomic-embed-text")



async def process_batches(project_data,vector_store, batch_size=100):
    print("Processing batches")
    texts = project_data['cfAbstr'].tolist()
    titles = project_data['cfTitle'].tolist()
    combined = [title + " " + text for title, text in zip(titles, texts)]
    doc_ids = project_data['cfProjId'].tolist()

    for i in range(0, len(combined), batch_size * 5):  # Process in groups of 5 batches
        print(f"Processing batch {i // batch_size} to {i // batch_size + 5}")
        sub_tasks = [
            asyncio.create_task(vector_store.aadd_texts(
                texts=combined[j:j+batch_size], ids=doc_ids[j:j+batch_size]
            ))
            for j in range(i, min(i + batch_size * 5, len(combined)), batch_size)
        ]
        await asyncio.gather(*sub_tasks)

def convertToFaiss():
    vector_store = Chroma(embedding_function=embeddings,persist_directory = "data_projects_2024_5_vector_store_TitleAbstract")

    texts = []  
    vectors = []  
    ids = []
    # Iterate over all documents in Chroma
    for doc in vector_store.get()['documents']:
        texts.append(doc)
    for embedding in vector_store.get(include=['embeddings'])['embeddings']:
        vectors.append(embedding)
    for id in vector_store.get()['ids']:
        ids.append(id)

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    for i in range(0, len(texts), 1000):
        print(f"Processing batch {i // 1000}")
        text_embeddings = zip(texts[i:i+1000], vectors[i:i+1000])
        faiss_store = FAISS.from_embeddings(text_embeddings,embedding_model,metadatas=[{"doc_id": doc_id} for doc_id in ids[i:i+1000]])

    faiss_store.save_local("data_projects_2024_5_vector_store_TitleAbstract_faiss")

convertToFaiss()

if not os.path.exists("data_projects_2024_5_vector_store_TitleAbstract"):

    vector_store = Chroma(embedding_function=embeddings,persist_directory = "data_projects_2024_5_vector_store_TitleAbstract")

    asyncio.run(process_batches(project_data,vector_store,100))
    print("All tasks completed")

else:
    vector_store = Chroma(embedding_function=embeddings,persist_directory = "data_projects_2024_5_vector_store_TitleAbstract")





#print(vector_store.similarity_search("Rotavirus (RV) is the leading cause of severe gastroenteritis in children under 5 years of age. In this chapter we will document the burden of rotavirus gastroenteritis in the 27 countries of the European Union. Data on rotavirus disease burden is lacking for about half of the countries of the European Union. The available data for the other half show differences that do not necessarily reflect country-specific differences, but that may also be due to differences in study characteristics and quality. Based on the available literature, the annual incidence of RV disease in the European Union ranges from 0.064.15% of the children aged <5 years seeking ambulatory care, 02.65% of the children aged <5 years visiting emergency departments, 0.031.19% of the children aged <4/5 years being hospitalized for rotavirus, and 0.514 children per million children aged <3/5 years that die due to RV infection. Information on children experiencing disease but not seeking professional medical care is lacking for the European Union. Studies on nosocomial RV infections are numerous, but difficult (if not impossible) to interpret and compare, because of very different methodologies and contexts.", 2))

#embedAbstrs = embeddings.embed_documents(project_data['cfAbstr'].fillna("").tolist()[:1000])

# Add the embeddings to the dataframe
#project_data['embeddings'] = embedAbstrs
