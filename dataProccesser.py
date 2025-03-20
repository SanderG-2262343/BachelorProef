import pandas as pd
import xml.etree.ElementTree as ET
import glob
from bs4 import BeautifulSoup
import re
from bs4 import MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
import numpy as np
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import multiprocessing


namespaces = {'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
            'fris': 'http://fris.ewi.be/'}

def cleanUpProjectData():
    df = pd.read_csv('data/csvs/data_projects_2024_5_participants.csv')
    df = df[df['cfAbstr'].str.len() >= 50]
    df = df[df['cfAbstr'] != "A BOF-ZAP professorship granted by the Special Research Fund is a primarily research-oriented position and is made available for excellent researchers with a high-quality research programme."]
    df = df[df['cfAbstr'] != "A BOF-TT mandate holder receives an appointment as a Tenure Track with mainly research assignment. The salary costs are charged to the Special research Fund (BOF)."]
    df.drop_duplicates(subset=['cfAbstr'], inplace=True)
    df.to_csv('data/csvs/data_projects_2024_5_participants_noDuplicates.csv', index=False)




def extractTextFromHtml(html):
    if html is None:
        return ''
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator=" ")
    return re.sub(r'\s+', ' ', text).strip()


def getParticipants(participants):
    names = []
    for participant in participants:
        try:
            names.append(participant.find('cfFirstNames').text + " " + participant.find('cfFamilyNames').text)
        except:
            pass
    return ",".join(names)

def extractProjectsToCSV():
    df = pd.DataFrame()

    for xml_file in glob.glob("data/rawXml/data_projects_2024_5/*.xml"):
        print(f"Processing file: {xml_file}")
        tree = ET.parse(xml_file)

        root = tree.getroot()

        projects = root.findall('.//cfProj')


        rows = []
        for project in projects:
            try:
                if len(extractTextFromHtml(project.find('cfAbstr[@cfLangCode="en"]').text)) < 10:
                    #print(f"Empty abstract for project: {project.find('cfProjId').text}")
                    continue
                
                participants = project.findall('.//frParticipant')

                
                rows.append({   
                'cfProjId': project.find('cfProjId').text,
                'cfTitle': extractTextFromHtml(project.find('cfTitle[@cfLangCode="en"]').text),
                'cfAbstr': extractTextFromHtml(project.find('cfAbstr[@cfLangCode="en"]').text),
                'cfParticipants': getParticipants(participants)
                })
            except:   # if any of the fields are missing, skip this project
                pass
            

        temp_df = pd.DataFrame(rows)
        df = pd.concat([df, temp_df], ignore_index=True)

    df = df[df['cfAbstr'].str.len() >= 200]
    df = df[df['cfAbstr'] != "A BOF-ZAP professorship granted by the Special Research Fund is a primarily research-oriented position and is made available for excellent researchers with a high-quality research programme."]
    df = df[df['cfAbstr'] != "A BOF-TT mandate holder receives an appointment as a Tenure Track with mainly research assignment. The salary costs are charged to the Special research Fund (BOF)."]
    df.to_csv('data/csvs/data_projects_2024_5_particapants.csv', index=False)


def getProjectIds(publication):
    publ_ids = publication.findall('cfProj_ResPubl')
    projectIds = []
    for publ_id in publ_ids:
        projectIds.append(publ_id.find('cfProjId').text)
    output =  ",".join(projectIds) if projectIds else None 
    return output

def extractPublicationsToCSV():
    df = pd.DataFrame()

    

    for xml_file in glob.glob("data/rawXml/data_publications_2024_5/*.xml"):
        print(f"Processing file: {xml_file}")
        tree = ET.parse(xml_file)

        root = tree.getroot()

        publications = root.findall('.//cfResPubl')
        rows = []
        for publication in publications:
            try:
                rows.append({
                'cfPublId': publication.find('cfResPublId').text,
                'cfProjId': getProjectIds(publication),
                'cfTitle': extractTextFromHtml(publication.find('cfTitle[@cfLangCode="en"]').text),
                'cfAbstr': extractTextFromHtml(publication.find('cfAbstr[@cfLangCode="en"]').text),
                })
            except AttributeError as e:
                #
                #if publication.find('cfResPublId') is None:
                #    print(f"Publication has no ID")
                #elif publication.find('cfTitle[@cfLangCode="en"]') is None:
                #    print(f"Publication has no english title")
                #elif publication.find('cfAbstr[@cfLangCode="en"]') is None:
                #    print(f"Publication has no english abstract")
                #else:
                #    print(f"Unknown error: {e}")
                pass

        temp_df = pd.DataFrame(rows)
        df = pd.concat([df, temp_df], ignore_index=True)

    
    df.to_csv('data/csvs/data_publications_2024_5_2.csv', index=False)
    

def getParticipantsFris(project):
    participants = project.findall('.//fris:participants/fris:participant',namespaces)   

    participantsList = []
    for participant in participants:
        try:
            name = participant.find('./fris:assignment/fris:person/fris:name',namespaces)
            participantsList.append(name.find('./fris:firstName',namespaces).text + " " + name.find('./fris:lastName',namespaces).text)
        except AttributeError as e:  # if any of the fields are missing, skip this participant
            pass
    return ", ".join(participantsList)

def getDisciplinesFris(project, flemish = False):
    if flemish:
        disciplines = project.findall('.//fris:flemishDisciplines/fris:flemishDisciplines',namespaces)
    else:
        disciplines = project.findall('.//fris:disciplines/fris:discipline',namespaces)

    disciplinesList = []
    for discipline in disciplines:
        try:
            if flemish:
                disciplinesList.append(discipline.attrib['term'] +":" + discipline.find('./fris:description/fris:texts/fris:text[@locale="en"]',namespaces).text)
            else:
                disciplinesList.append(discipline.find('./fris:description/fris:texts/fris:text[@locale="en"]',namespaces).text)
        except:
            pass
    return ";".join(disciplinesList)


def getProjectIdsFris(publication):
    projects = publication.findall('./fris:researchOutputProjects/fris:researchOutputProject/fris:project',namespaces)
    projectIds = []
    for project in projects:
        projectIds.append(project.attrib['uuid'])
    return  ",".join(projectIds)

def getOrganisations(project):
    organizations = project.findall('.//fris:organisation/fris:name/fris:texts/fris:text[@locale="en"]',namespaces)
    if organizations is None:
        return ""
    organizationsList = []
    for organization in organizations:
        try:
            if organization.text:
                organizationsList.append(organization.text)
        except AttributeError as e:
            pass
    return ", ".join(set(organizationsList))
    
# Function to extract projects from FRIS to CSV with newer format
def extractProjectsToCSVFris():
    
    df = pd.DataFrame()

    for xml_file in glob.glob("data/rawXml/data_projects_2024_5_2/*.xml"):
        print(f"Processing file: {xml_file}")
        tree = ET.parse(xml_file)

        root = tree.getroot()

        projects = root.findall('.//fris:project', namespaces)


        rows = []
        for project in projects:
            try:
                dataProvider = project.find("./fris:dataProvider",namespaces)             
                rows.append({   
                'projId': project.attrib['uuid'],
                'title': extractTextFromHtml(project.find('./fris:name/fris:texts/fris:text[@locale="en"]',namespaces).text),
                'abstract': extractTextFromHtml(project.find('./fris:projectAbstract/fris:texts/fris:text[@locale="en"]',namespaces).text),
                'participants': getParticipantsFris(project),
                'disciplines':  getDisciplinesFris(project),
                'flemishDisciplines': getDisciplinesFris(project,flemish=True),
                'organization': getOrganisations(project),
                'dataProvider': dataProvider.text if dataProvider is not None and dataProvider.text is not None  else ""
                })
            except AttributeError as e:
                pass
        temp_df = pd.DataFrame(rows)
        df = pd.concat([df, temp_df], ignore_index=True)
        

    df = df[df['abstract'].str.len() >= 150] # remove projects with short abstracts
    df = df[df['title'].str.len() >= 5] # remove projects with no titles

    
    df.to_csv('data/csvs/data_projects_2024_5_FRIS_2.csv', index=False)


def getDfFromPublicationXml(xml_file):
        with open(xml_file, "r", encoding="utf-8", errors="replace") as f:
            xml_content = f.read() 
        xml_content = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u0080-\uFFFF]', '', xml_content)
        tree = ET.ElementTree(ET.fromstring(xml_content))

        root = tree.getroot()

        #publications = root.findall('.//fris:contributionToConference', namespaces)
        publications = root.findall('.//fris:journalContribution', namespaces)

        rows = []
        for publication in publications:
            try:
                dataProvider = publication.find("./fris:dataProvider",namespaces)             
                rows.append({   
                'id': publication.attrib['uuid'],
                'projId': getProjectIdsFris(publication),
                'title': extractTextFromHtml(publication.find('./fris:title/fris:texts/fris:text[@locale="en"]',namespaces).text),
                'abstract': extractTextFromHtml(publication.find('./fris:researchAbstract/fris:texts/fris:text[@locale="en"]',namespaces).text),
                'participants': getParticipantsFris(publication),
                'disciplines':  getDisciplinesFris(publication),
                'organization': getOrganisations(publication),
                'flemishDisciplines': getDisciplinesFris(publication,flemish=True),
                'dataProvider': dataProvider.text if dataProvider is not None and dataProvider.text is not None  else ""
                })
            except AttributeError as e:
                pass

        temp_df = pd.DataFrame(rows)
        return temp_df
 
def extractPublicationsToCSVFris():
    df = pd.DataFrame()

    xml_files = glob.glob("data/rawXml/data_publications_2024_5/*.xml")
        
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(getDfFromPublicationXml, xml_files)


    for temp_df in results:

        df = pd.concat([df, temp_df], ignore_index=True)

    df = df[df['abstract'].str.len() >= 50] # remove projects with short abstracts
    df = df[df['title'].str.len() >= 5] # remove projects with no titles

    # remove projects for profferships and tenure tracks
    
    df.to_csv('data/csvs/data_publications_2024_5_FRIS.csv', index=False)


def getSimilarTestData():
    df = pd.read_csv('data/csvs/data_publications_2024_5.csv')
    df2 = pd.read_csv('data/csvs/data_publications_2024_5_FRIS.csv',dtype={'participants': str, 'disciplines': str, 'flemishDisciplines': str})

    df.dropna(how='any', inplace=True)   #Remove any with no projID matched

    df_merged = df.merge(df2[['id','participants']], left_on='cfPublId', right_on='id', how='left')
    df_merged.drop(columns=['id'],inplace=True)
    df_merged.to_csv('data/csvs/data_publications_2024_5_FRIS_matched.csv', index=False)


def mapVectorStore(vector_store):

    embeddings = vector_store.get(include=['embeddings'])['embeddings']
    text_labels = vector_store.get(include=['documents'])['documents']

    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    reduced_embeddings = reducer.fit_transform(embeddings)


    df = pd.DataFrame({
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "label": text_labels
    })
    df["label"] = df["label"].astype(str)
    fig = px.scatter(df, x="x", y="y", hover_data="label", title="UMAP Projection with Labels")
    fig.update_traces(textposition="top center")
    fig.show()

# didnt work
def createNormalized(vector_store_location):

    vector_store = Chroma(persist_directory = vector_store_location)
    data = vector_store.get(include=['documents','embeddings'])
    vector_store = Chroma(persist_directory = vector_store_location + "_normalized")
    data['embeddings'] = data['embeddings']/ np.linalg.norm(data['embeddings'],axis=1,keepdims=True)
    for i in range(0,len(data['documents']),1000):
        print(f"Processing batch {i}")
        vector_store._collection.add(documents=data['documents'][i:i+1000], ids=data['ids'][i:i+1000],embeddings=data['embeddings'][i:i+1000])
    
    return vector_store

def createTestSample():
    df = pd.read_csv('data/csvs/data_publications_2024_5_FRIS_WithProjIdsOnly.csv')
    df_proj = pd.read_csv('data/csvs/data_projects_2024_5_FRIS_2.csv')
    df_proj = df_proj[df_proj['abstract'].str.len() > 200]
    #df_sample = df.sample(100)
    #df_sample = pd.read_csv('data/csvs/data_publications_2024_5_TestSample.csv')
    projIds = df['projId'].str.split(',').explode().unique().tolist()
    df_proj = df_proj[df_proj['projId'].isin(projIds)]
    df = df[df['projId'].apply(lambda x: any(proj in x.split(',') for proj in df_proj['projId']))]
    df.to_csv('data/csvs/data_publications_2024_5_TestSample_Large.csv', index=False)
    df_proj.to_csv('data/csvs/data_projects_2024_5_TestSample_Large.csv', index=False)

#extractProjectsToCSV()
#extractPublicationsToCSV()
#extractProjectsToCSVFris()
if __name__ == "__main__":
    extractPublicationsToCSVFris()
    #extractProjectsToCSVFris()
#getSimilarTestData()
    #mapVectorStore(Chroma(persist_directory = "data/vectorStores/data_projects_2024_5_vector_store_VoyageAI_TestSample"))
    #createTestSample()
    
    
    
#createNormalized("data/vectorStores/data_projects_2024_5_vector_store_VoyageAI")
#cleanUpProjectData()
#mapVectorStore(Chroma(persist_directory = "data/vectorStores/data_projects_2024_5_vector_store_VoyageAI"))
#cleanUpProjectData()