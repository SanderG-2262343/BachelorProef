import pandas as pd
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from embeddingNomicLocal import process_batches
import asyncio
import time
import os
from dotenv import load_dotenv


if not os.path.exists("data_projects_2024_5_vector_store_VoyageAI"):
    print("Data already processed")
    exit()

load_dotenv()

# Load the data assuming the data is already preprocessed
project_data = pd.read_csv('data_projects_2024_5.csv')

# Drop rows with empty abstracts
project_data = project_data[project_data['cfAbstr'].str.len() >= 20] 
project_data.dropna(how='any', inplace=True) #one project with no title

embeddings = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])

vector_store = Chroma(embedding_function=embeddings,persist_directory = "data_projects_2024_5_vector_store_VoyageAI")


#4.9mil
asyncio.run(process_batches(project_data,vector_store,1000))

#batch_size = 128
#texts = project_data['cfAbstr'].tolist()
#titles = project_data['cfTitle'].tolist()
#combined = [title + " " + text for title, text in zip(titles, texts)]
#for i in range(0,len(project_data['cfAbstr']),batch_size):
#    print(f"Processing batch {i}")
#    vector_store.add_texts(texts=combined[i:i+batch_size], ids=project_data['cfProjId'][i:i+batch_size])
