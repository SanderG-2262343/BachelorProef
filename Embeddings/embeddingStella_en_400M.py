import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from embeddingNomicLocal import process_batches
import asyncio
import os



# Too slow to run on my GPU only got batch 0-5 done in 1 hour and a half


embeddings = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_400M_v5",    model_kwargs={"trust_remote_code": True,"device" :"cuda"})
project_data = pd.read_csv('data_projects_2024_5.csv')
project_data = project_data[project_data['cfAbstr'].str.len() >= 20] 
project_data.dropna(how='any', inplace=True) #one project with no abstract


if not os.path.exists("data_projects_2024_5_vector_store_stella"): #If data not already processed
    vector_store = Chroma(embedding_function=embeddings,persist_directory="data_projects_2024_5_vector_store_stella")
    asyncio.run(process_batches(project_data, vector_store, batch_size=100))
else:
    print("Data already processed")