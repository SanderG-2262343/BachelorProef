from dataValidation import runAllTests,runTestsVoyageAi
from Embeddings.embeddingVoyageAI import voyageAIEmbedding
from Embeddings.embeddingNomicLocal import nomicEmbedding
from Embeddings.embeddingTop2Vec import top2vecModelTrain
from Embeddings.embeddingGemini import geminiEmbedding
from dataProccesser import createTestSample
import os
import shutil
import pandas as pd
#import nltk
#from nltk.corpus import stopwords
#nltk.download('stopwords')

#def text)
#    stop_words = set(stopwords.words('english'))
#    word_tokens = text.split(" ")
#    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#    return ' '.join(filtered_sentence)

def multipleSamples():
    
    for i in range(1, 10):
        testSampleLocation = f"TestSampleM_{i}.csv"
        if os.path.exists("data/csvs/data_publications_2024_5_" + testSampleLocation):
            os.remove("data/csvs/data_publications_2024_5_" + testSampleLocation)
            os.remove(f"data/csvs/data_projects_2024_5_TestSampleM_{i}.csv")
        if os.path.exists(f"data/embeddingsSaves/embeddingsVoyage_TestSampleM_{i}.csv"):
            os.remove(f"data/embeddingSaves/embeddingsVoyage_TestSampleM_{i}.csv")
        if os.path.exists(f"data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TestSampleM_{i}"):
            shutil.rmtree(f"data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TestSampleM_{i}")
        print(f"Creating test sample {i}")

        createTestSample(testSampleLocation)
        vector_storeVoyageAIPath = f"data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TestSampleM_{i}"
        voyageAIEmbedding(vector_storeVoyageAIPath,f"data/csvs/data_projects_2024_5_TestSampleM_{i}.csv")
        publicationdata = pd.read_csv("data/csvs/data_publications_2024_5_" + testSampleLocation)
        runTestsVoyageAi(publicationdata,vector_storeVoyageAIPath, f"data/embeddingSaves/embeddingsVoyage_TestSampleM_{i}.csv")
        


def main():
    vector_storeVoyageAIPath = "data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TA_Names_Disciplines_DP_TestSample"
    vector_storeNomicPath = "data/vectorStores/data_projects_2024_5_vector_store_Nomic_Names_TestSample"
    vector_storeGeminiPath = "data/vectorStores/data_projects_2024_5_vector_store_Gemini_exp_TestSample"
    top2vecModelPath = "data/models/top2vec_model_TestSample_Attr"
    currentDataset = "data/csvs/data_projects_2024_5_TestSample.csv"

    #geminiEmbedding(vector_storeGeminiPath,currentDataset)
    voyageAIEmbedding(vector_storeVoyageAIPath,currentDataset,zipfunction = lambda titles,abstracts,participants,disciplines,dataProviders: [f"Title: {title} Abstract: {abstract} Disciplines: {discipline}"  for title, abstract,discipline in zip(titles, abstracts,disciplines)])
    nomicEmbedding(vector_storeNomicPath,currentDataset,lambda titles,abstracts: ["Title: " + title + " Abstract:" + abstract for title, abstract in zip(titles, abstracts)])
    top2vecModelTrain(top2vecModelPath,currentDataset,lambda titles,abstracts: ["Title: " + title + " Abstract:" + abstract for title, abstract in zip(titles, abstracts)])
    runAllTests(vector_storeVoyageAIPath, "data/embeddingSaves/embeddingsVoyage_TA_Names_DP_Disc_TestSample.csv",
                vector_storeNomicPath, "data/embeddingSaves/embeddingsNomicTest_Attr.csv",
                top2vecModelPath,"data/csvs/data_publications_2024_5_TestSample_dataP.csv"
                ,[lambda titles,abstracts,participants,disciplines,dataProviders: ["Instruct: Compare this publication with a project \n Query: Title: " + title + " Abstract: " + abstract for title, abstract,dataProvider in zip(titles, abstracts,dataProviders)],lambda titles, abstracts: ["Title: " + title + " Abstract:" + abstract for title, abstract in zip(titles, abstracts)],lambda titles, abstracts: ["Title: " + title + " Abstract:" + abstract for title, abstract in zip(titles, abstracts)]]
                )
#main()
multipleSamples()

