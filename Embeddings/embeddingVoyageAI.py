import pandas as pd
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from embeddingNomicLocal import process_batches
import asyncio
import time
import os
from dotenv import load_dotenv

STORAGE_DIR = "data_projects_2024_5_vector_store_VoyageAI_Abstract"
CURRENT_DATASET = "data_projects_2024_5_participants.csv"

if os.path.exists(STORAGE_DIR): #If data not already processed
    print("Data already processed")
    exit()

load_dotenv()

# Load the data assuming the data is already preprocessed
project_data = pd.read_csv(f"data/csvs/{CURRENT_DATASET}")

# Drop rows with empty abstracts

#project_data = project_data[project_data['cfAbstr'].str.len() >= 20] 
#project_data.dropna(how='any', inplace=True) #one project with no title

embeddings = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])
embeddings.batch_size = 128

vector_store = Chroma(embedding_function=embeddings,persist_directory = "data/vectorStores/" + STORAGE_DIR)




#asyncio.run(process_batches(project_data,vector_store,1000))

batch_size = 1000
texts = project_data['cfAbstract'].tolist()
titles = project_data['CfTitle'].tolist()
#participants = project_data['participants'].tolist()
#combined = [f"Title: {title} Participants: {participants} Abstract: {text}" for title, text,participants in zip(titles, texts,participants)]
for i in range(0,len(project_data['cfAbstract']),batch_size):
    print(f"Processing batch {i}")
    vector_store.add_texts(texts=texts[i:i+batch_size], ids=project_data['cfProjId'][i:i+batch_size])
