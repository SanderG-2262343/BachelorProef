import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from embeddingVoyageAI import cleanParticipants, cleanDisciplines
import os

"""
    * creates embeddings using the LinqMinstral model and stores them in storageDir.
    * @param storageDir: The directory where the embeddings will be stored.
    * @param current_Dataset: The path to the dataset file.
    * @param zipfunction: A function that combines the title, abstract, participants, disciplines, and data provider into a single string for each project to embed.
    * @return: None
"""
def LinqMinstralEmbedding(storageDir,current_Dataset,
                      zipfunction = lambda titles,abstracts,participants,disciplines,dataProvider: [title + " " + abstract for title, abstract in zip(titles, abstracts)]):
    
    
    if os.path.exists(storageDir):
        print("Data already processed")
        return
    
    # Load the data assuming the data is already preprocessed
    project_data = pd.read_csv(current_Dataset)

    embeddings = HuggingFaceEmbeddings(model_name="LinqMinstral-7B-v1.0")
    
    vector_store = Chroma(embedding_function=embeddings,persist_directory = storageDir)


    batch_size = 100

    abstracts = project_data['abstract'].tolist()
    titles = project_data['title'].tolist()
    participants = project_data['participants'].tolist()
    participants = [cleanParticipants(participant) for participant in participants] # remove duplicate authors
    disciplines = project_data['flemishDisciplines'].tolist()
    disciplines = [cleanDisciplines(discipline) for discipline in disciplines] #remove prefix code
    dataProvider = project_data['dataProvider'].tolist()
    

    combined = zipfunction(titles,abstracts,participants,disciplines,dataProvider)
    for i in range(0,len(project_data['abstract']),batch_size):
        print(f"Processing batch {i}")
        vector_store.add_texts(texts=combined[i:i+batch_size], ids=project_data['projId'][i:i+batch_size],metadatas=[{"dataProvider": dp,"participants" : authors} for dp,authors in zip(dataProvider[i:i+batch_size],participants[i:i+batch_size])])

