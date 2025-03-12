from google import genai
from google.genai import types
import pandas as pd
from langchain_chroma import Chroma
import os
import time
from dotenv import load_dotenv
load_dotenv()


def geminiEmbedding(StorageDir,Current_Dataset,
                      zipfunction = lambda titles,abstracts,participants,disciplines: [title + " " + abstract for title, abstract in zip(titles, abstracts)]):

    if os.path.exists(StorageDir): #If data not already processed
        print("Data already processed")
        return
    
    df = pd.read_csv(Current_Dataset)

    combined = zipfunction(df['title'],df['abstract'],df['participants'],df['flemishDisciplines']) #DONT FORGET TO CLEAN PARTICIPANTS AND DISCIPLINES BEFORE ACTUALLY USING

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    vector_store = Chroma(embedding_function=client,persist_directory = StorageDir)
    for i in range(0,len(df['abstract']),5):
        print(f"Processing batch {i}")
        
        result = client.models.embed_content(model="gemini-embedding-exp-03-07",contents=combined[i:i+5])#,config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
        embeddings = [embedding.values for embedding in result.embeddings]
        vector_store._collection.upsert(ids=df["projId"][i:i+5].tolist(), embeddings=embeddings,documents=combined[i:i+5])
        time.sleep(60)