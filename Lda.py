from gensim import corpora, models
import langchain_chroma as chroma
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = text.split(" ")
    filtered_sentence = [w.lower() for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)

def sparse_to_dense(vector):
    dense = np.zeros(41)
    for index, value in vector:
        dense[index] = value
    return dense.tolist()


# Load the data assuming the data is already preprocessed
df = pd.read_csv("data/csvs/data_projects_2024_5_TestSample.csv")
corpus = [title + " " + abstract for title, abstract in zip(df['title'],df['abstract'])]
texts = [clean_text(doc).split(" ") for doc in corpus]
dictionary = corpora.Dictionary(texts)
corpusBow = [dictionary.doc2bow(text) for text in texts]
ldamodel = models.LdaModel(corpusBow, num_topics=41, id2word = dictionary, passes=1000,random_state=42)


#get the document vectors
doc_vectors = [ldamodel[doc] for doc in corpusBow]

doc_vectors = [sparse_to_dense(doc) for doc in doc_vectors]


# Create a Chroma vector store and add the document vectors
vector_store = chroma.Chroma()
for i in range(0,len(doc_vectors),100):
    vector_store._collection.upsert(ids=df['projId'][i:i + 100].tolist(), embeddings=doc_vectors[i:i + 100],documents=corpus[i:i + 100])

# Evaluate the model
df2 = pd.read_csv("data/csvs/data_publications_2024_5_TestSample_dataP.csv")
for k in [1,2,3,5] + list(range(10, 110, 10)):
    sucessmatch = 0
    for i in range(0,len(df2['abstract'])):
        bow = dictionary.doc2bow(clean_text(df2['title'][i] + " " + df2['abstract'][i]).split(" "))
        vec = ldamodel.get_document_topics(bow)

        vec = sparse_to_dense(vec)
        results = vector_store.similarity_search_by_vector(vec,k=k)

        for result in results:
            if result.id in df2['projId'][i]:
                sucessmatch += 1
                break
    print(f"Top k: {k} Result: {sucessmatch * 100/len(df2['abstract'])}")