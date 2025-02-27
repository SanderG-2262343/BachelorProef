import pandas as pd
import xml.etree.ElementTree as ET
import glob
from bs4 import BeautifulSoup
import re


def cleanUpProjectData():
    df = pd.read_csv('data_projects_2024_5.csv')
    df = df[df['cfAbstr'].str.len() >= 50]
    df = df[df['cfAbstr'] != "A BOF-ZAP professorship granted by the Special Research Fund is a primarily research-oriented position and is made available for excellent researchers with a high-quality research programme."]
    df = df[df['cfAbstr'] != "A BOF-TT mandate holder receives an appointment as a Tenure Track with mainly research assignment. The salary costs are charged to the Special research Fund (BOF)."]
    df.to_csv('data_projects_2024_5_clean.csv', index=False)



def extractTextFromHtml(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator=" ")
    return re.sub(r'\s+', ' ', text).strip()


def getParticapants(participants):
    names = []
    for participant in participants:
        try:
            names.append(participant.find('cfFirstNames').text + " " + participant.find('cfFamilyNames').text)
        except:
            pass
    return ",".join(names)

def extractProjectsToCSV():
    df = pd.DataFrame()

    for xml_file in glob.glob("data_projects_2024_5/*.xml"):
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
                'cfParticipants': getParticapants(participants)
                })
            except:   # if any of the fields are missing, skip this project
                pass
            

        temp_df = pd.DataFrame(rows)
        df = pd.concat([df, temp_df], ignore_index=True)

    df = df[df['cfAbstr'].str.len() >= 50]
    df = df[df['cfAbstr'] != "A BOF-ZAP professorship granted by the Special Research Fund is a primarily research-oriented position and is made available for excellent researchers with a high-quality research programme."]
    df = df[df['cfAbstr'] != "A BOF-TT mandate holder receives an appointment as a Tenure Track with mainly research assignment. The salary costs are charged to the Special research Fund (BOF)."]
    df.to_csv('data_projects_2024_5_particapants.csv', index=False)


def getProjectIds(publication):
    publ_ids = publication.findall('cfProj_ResPubl')
    projectIds = []
    for publ_id in publ_ids:
        projectIds.append(publ_id.find('cfProjId').text)
    output =  ",".join(projectIds) if projectIds else None 
    return output

def extractPublicationsToCSV():
    df = pd.DataFrame()

    for xml_file in glob.glob("data_publications_2024_5/*.xml"):
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

    
    df.to_csv('data_publications_2024_5_2.csv', index=False)


extractProjectsToCSV()
#extractPublicationsToCSV()


#cleanUpProjectData()