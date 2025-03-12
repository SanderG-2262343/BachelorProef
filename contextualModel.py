from bertopic import BERTopic
from langchain_chroma import Chroma
import pandas as pd


vector_database = Chroma(persist_directory="data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TestSample")
df = pd.read_csv("data/csvs/data_projects_2024_5_TestSample.csv")

#embeddings = vector_database.get(include=['embeddings'])['embeddings']
model = BERTopic(language="english",calculate_probabilities=True)
combined = [f"{title} {abstract}" for title,abstract in zip(df['title'],df['abstract'])]
model.fit(combined)
model.visualize_topics()
print(model.get_topics())
print(model.find_topics("cancer",5))



