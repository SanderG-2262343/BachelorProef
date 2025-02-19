import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import asyncio
import os

# Load the data assuming the data is already preprocessed
project_data = pd.read_csv('data_projects_2024_5.csv')

# Initialize the OllamaEmbeddings object assume the model already pulled
embeddings = OllamaEmbeddings(model="nomic-embed-text")



async def process_batches():
    batch_size = 100
    texts = project_data['cfAbstr'].tolist()
    doc_ids = project_data['cfProjId'].tolist()

    for i in range(0, len(texts), batch_size * 5):  # Process in groups of 5 batches
        print(f"Processing batch {i // batch_size} to {i // batch_size + 5}")
        sub_tasks = [
            asyncio.create_task(vector_store.aadd_texts(
                texts=texts[j:j+batch_size], ids=doc_ids[j:j+batch_size]
            ))
            for j in range(i, min(i + batch_size * 5, len(texts)), batch_size)
        ]
        await asyncio.gather(*sub_tasks)


if not os.path.exists("data_projects_2024_5_vector_store"):

    vector_store = Chroma(embedding_function=embeddings,persist_directory = "data_projects_2024_5_vector_store")

    asyncio.run(process_batches())
    print("All tasks completed")

else:
    vector_store = Chroma(embedding_function=embeddings,persist_directory = "data_projects_2024_5_vector_store")



print(vector_store.similarity_search("Rotavirus (RV) is the leading cause of severe gastroenteritis in children under 5 years of age. In this chapter we will document the burden of rotavirus gastroenteritis in the 27 countries of the European Union. Data on rotavirus disease burden is lacking for about half of the countries of the European Union. The available data for the other half show differences that do not necessarily reflect country-specific differences, but that may also be due to differences in study characteristics and quality. Based on the available literature, the annual incidence of RV disease in the European Union ranges from 0.064.15% of the children aged <5 years seeking ambulatory care, 02.65% of the children aged <5 years visiting emergency departments, 0.031.19% of the children aged <4/5 years being hospitalized for rotavirus, and 0.514 children per million children aged <3/5 years that die due to RV infection. Information on children experiencing disease but not seeking professional medical care is lacking for the European Union. Studies on nosocomial RV infections are numerous, but difficult (if not impossible) to interpret and compare, because of very different methodologies and contexts.", 2))

#embedAbstrs = embeddings.embed_documents(project_data['cfAbstr'].fillna("").tolist()[:1000])

# Add the embeddings to the dataframe
#project_data['embeddings'] = embedAbstrs
