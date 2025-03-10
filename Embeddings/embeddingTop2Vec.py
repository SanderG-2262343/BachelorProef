import pandas as pd
import os
from top2vec import Top2Vec
#from langchain_chroma import Chroma
#from langchain_core.documents import Document


def top2vecModelTrain(modelFilePath,Current_Dataset,zipfunction = lambda titles,abstracts: [title + " " + abstract for title, abstract in zip(titles, abstracts)]):
    # Load the data assuming the data is already preprocessed
    df = pd.read_csv(Current_Dataset)

    #df.dropna(how='any', inplace=True)
    #df = df[df['cfAbstr'].str.len() >= 20]

    # Load your text data
    abstracts = df['abstract'].tolist()
    titles = df['title'].tolist()
    combined = zipfunction(titles,abstracts)




    # Train Top2Vec (can take time for large datasets)
    if os.path.exists(modelFilePath):
        top2vec_model = Top2Vec.load(modelFilePath)
    else:
        top2vec_model = Top2Vec(combined, speed="deep-learn", workers=16, document_ids=df['projId'].tolist(),min_count=10)
        top2vec_model.save(modelFilePath)

# Get document embeddings
#doc_vectors = top2vec_model.document_vectors
#doc_ids = top2vec_model.document_ids
#doc_ids = list(doc_ids)
#vector_store = Chroma(persist_directory="data_projects_2024_5_vector_store_top2vec_2")
#for i in range(0, len(doc_vectors), 1000):
#    print(f"Inserting embeddings {i} to {i+1000}")
#    vector_store._collection.upsert(ids=doc_ids[i:i+1000], embeddings=doc_vectors[i:i+1000],documents=texts[i:i+1000])
