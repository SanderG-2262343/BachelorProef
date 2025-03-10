from dataValidation import runAllTests 
from Embeddings.embeddingVoyageAI import voyageAIEmbedding
from Embeddings.embeddingNomicLocal import nomicEmbedding
from Embeddings.embeddingTop2Vec import top2vecModelTrain


def main():
    vector_storeVoyageAIPath = "data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TestSample_Disciplines"
    vector_storeNomicPath = "data/vectorStores/data_projects_2024_5_vector_store_Nomic_TestSample"
    top2vecModelPath = "data/models/top2vec_model_TestSample_min_count_10"
    currentDataset = "data/csvs/data_projects_2024_5_TestSample.csv"

    voyageAIEmbedding(vector_storeVoyageAIPath,currentDataset,zipfunction = lambda titles,abstracts,participants,disciplines: [title + " " + discipline + " " + abstract  for title, abstract,discipline in zip(titles, abstracts,disciplines)])
    nomicEmbedding(vector_storeNomicPath,currentDataset)
    top2vecModelTrain(top2vecModelPath,currentDataset)
    runAllTests(vector_storeVoyageAIPath, "data/embeddingSaves/embeddingsVoyageTest.csv",
                vector_storeNomicPath, "data/embeddingSaves/embeddingsNomicTest.csv",
                top2vecModelPath,"data/csvs/data_publications_2024_5_TestSample.csv")
main()
