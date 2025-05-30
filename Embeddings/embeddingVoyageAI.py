import pandas as pd
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv



def cleanParticipants(participants):
    if pd.isnull(participants):
        return ""
    return ", ".join(set(participants.split(", ")))

def cleanDisciplines(disciplines):
    if pd.isnull(disciplines):
        return ""
    set_disciplines = set([d[7:] for d in disciplines.split(";")])
    return ", ".join(set_disciplines)

STORAGE_DIR = "data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_Title"
CURRENT_DATASET = "data/csvs/data_projects_2024_5_participants.csv"
def voyageAIEmbedding(StorageDir = STORAGE_DIR,Current_Dataset = CURRENT_DATASET,
                      zipfunction = lambda titles,abstracts,participants,disciplines,dataProvider: [title + " " + abstract for title, abstract in zip(titles, abstracts)],publications = False):

    if os.path.exists(StorageDir): #If data not already processed
        print("Data already processed")
        return

    load_dotenv()

    # Load the data assuming the data is already preprocessed
    project_data = pd.read_csv(Current_Dataset)

    # Drop rows with empty abstracts

    #project_data = project_data[project_data['cfAbstr'].str.len() >= 20] 
    #project_data.dropna(how='any', inplace=True) #one project with no title

    embeddings = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])
    embeddings.batch_size = 128

    



    #asyncio.run(process_batches(project_data,vector_store,1000))
    batch_size = 1000
    abstracts = project_data['abstract'].tolist()
    titles = project_data['title'].tolist()
    participants = project_data['participants'].tolist()
    participants = [cleanParticipants(participant) for participant in participants] # remove duplicate authors and split so a list of authors since going in metadata
    disciplines = project_data['flemishDisciplines'].tolist()
    disciplines = [cleanDisciplines(discipline) for discipline in disciplines] #remove prefix code
    dataProvider = project_data['dataProvider'].tolist()

    #participants = project_data['participants'].tolist()
    vector_store = Chroma(embedding_function=embeddings,persist_directory = StorageDir)
    combined = zipfunction(titles,abstracts,participants,disciplines,dataProvider)
    for i in range(0,len(project_data['abstract']),batch_size):
        print(f"Processing batch {i}")
        if publications:
            vector_store.add_texts(texts=combined[i:i+batch_size], ids=project_data['id'][i:i+batch_size],metadatas=[{"dataProvider": dp,"participants" : authors} for dp,authors in zip(dataProvider[i:i+batch_size],participants[i:i+batch_size])])
        else:
            vector_store.add_texts(texts=combined[i:i+batch_size], ids=project_data['projId'][i:i+batch_size],metadatas=[{"dataProvider": dp,"participants" : authors} for dp,authors in zip(dataProvider[i:i+batch_size],participants[i:i+batch_size])])