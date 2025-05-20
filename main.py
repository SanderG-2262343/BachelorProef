from dataValidation import runAllTests,runTestsVoyageAi,runTestsLingMinstral
from Embeddings.embeddingVoyageAI import voyageAIEmbedding
from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
#from Embeddings.embeddingNomicLocal import nomicEmbedding
#from Embeddings.embeddingTop2Vec import top2vecModelTrain

#from Embeddings.embeddingGemini import geminiEmbedding
#from Embeddings.embeddingLinqMinstral import LinqMinstralEmbedding
#from dataProccesser import createTestSample
import os
from dotenv import load_dotenv
#import shutil
import pandas as pd
#import nltk
#from nltk.corpus import stopwords
#nltk.download('stopwords')

#def text)
#    stop_words = set(stopwords.words('english'))
#    word_tokens = text.split(" ")
#    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
#    return ' '.join(filtered_sentence)


# Create multiple test samples and evaluate them, used for testing variance in samples
def multipleSamples():
    
    for i in range(0, 20):
        testSampleLocation = f"TestSampleM_{i}.csv"
        #if os.path.exists("data/csvs/data_publications_2024_5_" + testSampleLocation):
        #    os.remove("data/csvs/data_publications_2024_5_" + testSampleLocation)
        #    os.remove(f"data/csvs/data_projects_2024_5_TestSampleM_{i}.csv")
        #if os.path.exists(f"data/embeddingSaves/embeddingsVoyage_TestSampleM_{i}.csv"):
        #    os.remove(f"data/embeddingSaves/embeddingsVoyage_TestSampleM_{i}.csv")
        #if os.path.exists(f"data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TestSampleM_{i}"):
        #    shutil.rmtree(f"data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TestSampleM_{i}")
        print(f"Creating test sample {i}")

        #createTestSample(testSampleLocation)
        vector_storeVoyageAIPath = f"data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TestSampleM_{i}"
        voyageAIEmbedding(vector_storeVoyageAIPath,f"data/csvs/data_projects_2024_5_TestSampleM_{i}.csv")
        publicationdata = pd.read_csv("data/csvs/data_publications_2024_5_" + testSampleLocation)
        runTestsVoyageAi(publicationdata,vector_storeVoyageAIPath, f"data/embeddingSaves/embeddingsVoyage_TestSampleM_{i}.csv")
        


#Run all tests on the VoyageAI and Nomic and top2vec embeddings, define the zip function for data of each embedding
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
#multipleSamples()
#LinqMinstralEmbedding("data/vectorStores/data_projects_2024_5_vector_store_LinqMinstral_Title","data/csvs/data_projects_2024_5_TestSample.csv")

#publicationdata = pd.read_csv("data/csvs/data_publications_2024_5_TestSample_dataP.csv")
#runTestsLingMinstral(publicationdata,"data/vectorStores/data_projects_2024_5_vector_store_LinqMinstral_Title","data/embeddingSaves/embeddingsLinqMinstral_TestSample.csv")

def cleanDisciplines(disciplines):
    if pd.isnull(disciplines):
        return ""
    set_disciplines = set([d[7:] for d in disciplines.split(";")])
    return ", ".join(set_disciplines)

def getCorrelatedPublicationData(projId,k):
    """
    Get the correlated publication data for a given project ID.
    :param projId: The project ID to search for.
    :param k: The number of correlated publications to retrieve.
    :return: A DataFrame containing the correlated publication data.
    """

    pubdata = "data/csvs/data_publications_2024_5_NoDupl.csv"

    if not os.path.exists(pubdata):
    
        Unfilteredpublicationdatacsv = "data/csvs/data_publications_2024_5_FRIS.csv"
        
        publicationdata = pd.read_csv(Unfilteredpublicationdatacsv)
        print(len(publicationdata))
        agg_funcs = {
        'id': 'first',
        'projId': lambda x: ', '.join(sorted(set(str(v) for v in x.dropna()))),
        'abstract': 'first',
        'participants': lambda x: ', '.join(sorted(set(sum([str(i).split(', ') for i in x.dropna()], [])))),
        'disciplines': lambda x: ', '.join(sorted(set(sum([str(i).split(', ') for i in x.dropna()], [])))),
        'organization': lambda x: ', '.join(sorted(set(sum([str(i).split(', ') for i in x.dropna()], [])))),
        'flemishDisciplines': lambda x: ', '.join(sorted(set(sum([str(i).split(', ') for i in x.dropna()], [])))),
        'dataProvider': lambda x: ', '.join(sorted(set(str(v) for v in x.dropna())))
        }

        # Group by title and aggregate
        merged_df = publicationdata.groupby('title', as_index=False).agg(agg_funcs)
        print(len(merged_df))

        # Save to a new CSV if needed
        merged_df.to_csv(pubdata, index=False)

    if not os.path.exists("data/vectorStores/data_projects_2024_5_vector_store_voyage_publications"):
        print("Making vector store")
        voyageAIEmbedding("data/vectorStores/data_projects_2024_5_vector_store_voyage_publications",pubdata,zipfunction = lambda titles,abstracts,participants,disciplines,dataProviders: [f"Title: {title} Abstract: {abstract}"  for title, abstract in zip(titles, abstracts)],publications = True)

    # Load the vector store
    vector_store = Chroma(persist_directory="data/vectorStores/data_projects_2024_5_vector_store_voyage_publications")

    load_dotenv()

    # Load the embeddings
    model = VoyageAIEmbeddings(model="voyage-3-large",api_key=os.environ['VOYAGE_API_KEY'])

    df = pd.read_csv("data/csvs/data_projects_2024_5_FRIS_2.csv")
    # Get the project data
    project_data = df[df['projId'] == projId]


    input = f"Instruct: Compare this project with a publication \n Query: Title: {project_data["title"].tolist()[0]} Abstract: {project_data["abstract"].tolist()[0]} Disciplines: {cleanDisciplines(project_data["flemishDisciplines"].tolist()[0])}" 
    embedding = model.embed_documents([input])

    result = vector_store.similarity_search_by_vector(embedding, k=k, 
                                                      #filter={"dataProvider": {"$contains": project_data["dataProvider"]}}
                                                      )

    return [result.page_content for result in result]  #convert to a list of page_content

#getCorrelatedPublicationData("915b7539-7c50-4c20-8f81-d7f437720871",5)
