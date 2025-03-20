from dataValidation import runAllTests 
from Embeddings.embeddingVoyageAI import voyageAIEmbedding
from Embeddings.embeddingNomicLocal import nomicEmbedding
from Embeddings.embeddingTop2Vec import top2vecModelTrain
from Embeddings.embeddingGemini import geminiEmbedding
#import nltk
#from nltk.corpus import stopwords
#nltk.download('stopwords')

#def text)
#    stop_words = set(stopwords.words('english'))
#    word_tokens = text.split(" ")
#    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#    return ' '.join(filtered_sentence)

def main():
    vector_storeVoyageAIPath = "data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TestSample_NoDuplDisciplines"
    vector_storeNomicPath = "data/vectorStores/data_projects_2024_5_vector_store_MxBai_TestSample"
    vector_storeGeminiPath = "data/vectorStores/data_projects_2024_5_vector_store_Gemini_exp_TestSample"
    top2vecModelPath = "data/models/top2vec_model_TestSample"
    currentDataset = "data/csvs/data_projects_2024_5_TestSample.csv"

    geminiEmbedding(vector_storeGeminiPath,currentDataset)
    #voyageAIEmbedding(vector_storeVoyageAIPath,currentDataset,zipfunction = lambda titles,abstracts,participants,disciplines: [f"{title} {discipline} {abstract}"  for title, abstract,discipline in zip(titles, abstracts,disciplines)])
    #nomicEmbedding(vector_storeNomicPath,currentDataset,lambda titles,abstracts: [title + " " + abstract for title, abstract in zip(titles, abstracts)])
    #top2vecModelTrain(top2vecModelPath,currentDataset,lambda titles,abstracts: [title + " " + abstract for title, abstract in zip(titles, abstracts)])
    #runAllTests(vector_storeVoyageAIPath, "data/embeddingSaves/embeddingsVoyageTest.csv",
    #            vector_storeNomicPath, "data/embeddingSaves/embeddingsMxBaiTest_Instruct.csv",
    #            top2vecModelPath,"data/csvs/data_publications_2024_5_TestSample.csv"
                #,[None,lambda titles,abstracts: ["Instruct: Compare this publication with a project \n Query:" + title + " " + abstract for title, abstract in zip(titles, abstracts)],None]
    #            )
main()
