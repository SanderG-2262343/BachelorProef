import zeep  # RC: needed to make wsdl soap requests. https://docs.python-zeep.org/en/master/
import datetime
import os
from lxml import etree
import re

parser = etree.XMLParser(recover=True)

PROJECTS = True

pageNumber = 0  # RC: pageNumber start value
pageSize = 1000  # RC: pageSize
previousUUID = '' # RC: needed to compare the UUID with the previous one. If it's the same, exit the program

xml = {
    "criteria": {
        "window": {
            "pageSize": "",
            "pageNumber": "",
            "orderings": {
                "order": {
                    "id": "entity.created",
                    "locale": "en",
                    "direction": "ASCENDING"
                }
            }
        },
        #"typeClassification": {
        #    "term": "Journal Contribution Type",
        #    "schemeId": "Research Output Taxonomy Type",
        #    "hierarchical": "true"
        #},
        #      "dataProviders": {
        #        "identifier": "UHasselt"
        #      },
        #       "search": {
        #            "search": "3e92f4a7-e717-4327-81a7-3dea37b7ac1f" ,
        #       }
    }
}

xml['criteria']['window']['pageNumber'] = pageNumber
xml['criteria']['window']['pageSize'] = pageSize
if not PROJECTS:
    xml['criteria'].update({"typeClassification": {
        "term": "Journal Contribution Type",
        "schemeId": "Research Output Taxonomy Type",
        "hierarchical": "true"
    }})

# RC: setup soap and do a request to get the total number of publications
# RC: test service    
# wsdl = 'https://app-acceptance.r4.researchportal.be/ws/ResearchOutputServiceFRIS?WSDL' 
# RC: production service
wsdl = 'https://frisr4.researchportal.be/ws/ResearchOutputServiceFRIS?WSDL'

if PROJECTS:
    wsdl = 'https://frisr4.researchportal.be/ws/ProjectServiceFRIS?WSDL'

# RC: define the namespaces, so we can use the shortcuts (soap, fris) in our xml parsing
namespaces = {'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
            'fris': 'http://fris.ewi.be/'}

# RC: initialise the zeep client
settings = zeep.Settings(strict=True, xml_huge_tree=True, raw_response=True)
client = zeep.Client(wsdl=wsdl, settings=settings)

# RC: to find the total amount of journals, we have to do a request and read the 'total' attribute of the queryResult tag
if PROJECTS:
    soapResult = client.service.getProjects(**xml)
else:
    soapResult = client.service.getResearchOutput(**xml)
    print(soapResult.content)
myroot = etree.fromstring(soapResult.content, parser=parser)


if PROJECTS:
    queryResult = myroot.find('./soap:Body/fris:getProjectsResponse/queryResult', namespaces)
else:
    queryResult = myroot.find('./soap:Body/fris:getResearchOutputResponse/queryResult', namespaces)

# RC: get the total number of requests
total = int(queryResult.attrib['total'])
#    total = 300 # RC: testing 

totalRequests = int(total / pageSize) + 1
print('Trying to download ' + str(total) + ' journal contributions.')

previousTime = datetime.datetime.now()
while pageNumber < totalRequests:
    isinstance(pageNumber, (int, float, complex)) and not isinstance(pageNumber, bool)
    # RC: change the pageNumber in the xml
    xml['criteria']['window']['pageNumber'] = pageNumber
    # RC: do the actual request
    if PROJECTS:
        soapResult = client.service.getProjects(**xml)
    else:
        soapResult = client.service.getResearchOutput(**xml)

    myroot = etree.fromstring(soapResult.content, parser=parser)


    if PROJECTS:
        os.makedirs('./data_projects_2024_5', exist_ok=True)
        with open('./data_projects_2024_5/page%s.xml' % pageNumber, 'w',encoding="utf-8") as f:
            clean_text = re.sub(r'[^\x20-\x7E\t\n\r]', '', soapResult.text)
            f.write(clean_text)
            f.close()
    else:
        os.makedirs('./data_publications_2024_5', exist_ok=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        destination_path = os.path.join(script_dir, "data_publications_2024_5/page%s.xml" % pageNumber)
        open(destination_path, 'wb').write(soapResult.content)
    

    
    timeDiff = datetime.datetime.now() - previousTime
    previousTime = datetime.datetime.now()        
    timeDiff = str(timeDiff.total_seconds())

    pageNumber = pageNumber + 1

    if PROJECTS:
        print('Page number: ' + str(pageNumber))
    else:

        # RC: find all the journals contributions in the soap result
        jrnlContribs = myroot.findall('./soap:Body/fris:getResearchOutputResponse/queryResult/fris:journalContribution', namespaces)

        # RC: initialise the arrays
        JRNL_CONTRIB_UUID = []

        teller = 0
        for jrnlContrib in jrnlContribs:
            # RC: set defaults
            JRNL_CONTRIB_UUID.append(None)

            # RC: get the fris ID
            if jrnlContrib.attrib['uuid'] is not None:
                JRNL_CONTRIB_UUID[teller] = jrnlContrib.attrib['uuid'][0:50]

            teller = teller + 1

        
        if len(JRNL_CONTRIB_UUID) == 0:
            pageNumber = pageNumber - 1
            JRNL_CONTRIB_UUID.append('N/A')

        output = str(teller * pageNumber) + ' - ' + JRNL_CONTRIB_UUID[0] + ' - ' + timeDiff + ' seconds.'
        print(output)

        # RC: check whether we receive the same chunck of data again 
        if JRNL_CONTRIB_UUID[0] != 'N/A':
            if JRNL_CONTRIB_UUID[0] == previousUUID:
                output = 'Doubles error: ' + JRNL_CONTRIB_UUID[0] + ' - ' + previousUUID
                print(output)
                exit()
            previousUUID = JRNL_CONTRIB_UUID[0]
    

print('Done ...')

