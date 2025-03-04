import pandas as pd
import xml.etree.ElementTree as ET
import glob
from bs4 import BeautifulSoup
import re

from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma

namespaces = {'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
            'fris': 'http://fris.ewi.be/'}

def cleanUpProjectData():
    df = pd.read_csv('data/csvs/data_projects_2024_5.csv')
    df = df[df['cfAbstr'].str.len() >= 50]
    df = df[df['cfAbstr'] != "A BOF-ZAP professorship granted by the Special Research Fund is a primarily research-oriented position and is made available for excellent researchers with a high-quality research programme."]
    df = df[df['cfAbstr'] != "A BOF-TT mandate holder receives an appointment as a Tenure Track with mainly research assignment. The salary costs are charged to the Special research Fund (BOF)."]
    df.to_csv('data/csvs/data_projects_2024_5_clean.csv', index=False)



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

    df = df[df['cfAbstr'].str.len() >= 50]
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
        if flemish:
            disciplinesList.append(discipline.attrib['term'] +":" + discipline.find('./fris:description/fris:texts/fris:text[@locale="en"]',namespaces).text)
        else:
            disciplinesList.append(discipline.find('./fris:description/fris:texts/fris:text[@locale="en"]',namespaces).text)
    return ";".join(disciplinesList)


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

                rows.append({   
                'projId': project.attrib['uuid'],
                'title': extractTextFromHtml(project.find('./fris:name/fris:texts/fris:text[@locale="en"]',namespaces).text),
                'abstract': extractTextFromHtml(project.find('./fris:projectAbstract/fris:texts/fris:text[@locale="en"]',namespaces).text),
                'participants': getParticipantsFris(project),
                'disciplines':  getDisciplinesFris(project),
                'flemishDisciplines': getDisciplinesFris(project,flemish=True)
                })
            except AttributeError as e:
                pass
        temp_df = pd.DataFrame(rows)
        df = pd.concat([df, temp_df], ignore_index=True)
        

    df = df[df['abstract'].str.len() >= 50] # remove projects with short abstracts
    df = df[df['title'].str.len() >= 5] # remove projects with no titles

    # remove projects for profferships and tenure tracks
    df = df[df['abstract'] != "A BOF-ZAP professorship granted by the Special Research Fund is a primarily research-oriented position and is made available for excellent researchers with a high-quality research programme."]
    df = df[df['abstract'] != "A BOF-TT mandate holder receives an appointment as a Tenure Track with mainly research assignment. The salary costs are charged to the Special research Fund (BOF)."]
    
    df.to_csv('data/csvs/data_projects_2024_5_FRIS.csv', index=False)


def extractPublicationsToCSVFris():
    df = pd.DataFrame()

    for xml_file in glob.glob("data/rawXml/data_publications_2024_5/*.xml"):
        print(f"Processing file: {xml_file}")

        with open(xml_file, "r", encoding="utf-8", errors="replace") as f:
            xml_content = f.read() 

        xml_content = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u0080-\uFFFF]', '', xml_content)
        tree = ET.ElementTree(ET.fromstring(xml_content))

        root = tree.getroot()

        publications = root.findall('.//fris:journalContribution', namespaces)

        rows = []
        for publication in publications:
            try:               

                rows.append({   
                'id': publication.attrib['uuid'],
                'title': extractTextFromHtml(publication.find('./fris:title/fris:texts/fris:text[@locale="en"]',namespaces).text),
                'abstract': extractTextFromHtml(publication.find('./fris:researchAbstract/fris:texts/fris:text[@locale="en"]',namespaces).text),
                'participants': getParticipantsFris(publication),
                'disciplines':  getDisciplinesFris(publication),
                'flemishDisciplines': getDisciplinesFris(publication,flemish=True)
                })
            except AttributeError as e:
                pass
        temp_df = pd.DataFrame(rows)
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


#extractProjectsToCSV()
#extractPublicationsToCSV()
#extractProjectsToCSVFris()
#extractPublicationsToCSVFris()
getSimilarTestData()


#cleanUpProjectData()