import pandas as pd
import xml.etree.ElementTree as ET


print("Start reading xml file")

tree = ET.parse('data_projects_2024_5/page0.xml')
root = tree.getroot()

projects = root.findall('.//cfProj')

rows = []
for project in projects:
    
    rows.append({
    'cfProjId': project.find('cfProjId').text,
    'cfTitle': project.find('cfTitle[@cfLangCode="en"]').text,
    'cfAbstr': project.find('cfAbstr[@cfLangCode="en"]').text
    })

df = pd.DataFrame(rows)

print(df.to_string(index=False))
