
import requests
import time
import os
import xml.etree.ElementTree as ET
import pandas as pd


def clean_cerif_xml_data(xmlstring):
    xmlstring = xmlstring.replace(' xmlns="http://fris.ewi.be/response"', '')
    xmlstring = xmlstring.replace(' xmlns="urn:xmlns:org:eurocris:cerif-1.5-1-FRIS"', '')
    xmlstring = xmlstring.replace(' xmlns="urn:xmlns:org:eurocris:cerif-1.5-1"', '')
    xmlstring = xmlstring.replace(' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"', '')
    xmlstring = xmlstring.replace(' xsi:nil="true"', '')
    return xmlstring

def fetch_from_service(url, headers, body, max_pages, destination):
    with requests.Session() as session:
        page = 0
        retries = 0 
        skippedpage = []
        while page < max_pages:
            body_for_this_page = body % page

            script_dir = os.path.dirname(os.path.abspath(__file__))
            destination_path = os.path.join(script_dir, destination % page)

            
            response = session.post(url, data=body_for_this_page, headers=headers)



            try:
                root = ET.fromstring(response.text)

                # Check if SOAP Fault retry
                fault_element = root.find('.//{http://schemas.xmlsoap.org/soap/envelope/}Fault')
                if fault_element is not None:
                    print(f"SOAP Fault detected on page {page}, retrying in 10 second...")
                    ET.dump(root)
                    retries += 1
                    time.sleep(10)
                    continue
                namespaces = {
                    'soap': "http://schemas.xmlsoap.org/soap/envelope/",
                    'ns1': "http://fris.ewi.be/",
                    'ns2': "http://fris.ewi.be/response",
                }
                # Check if CERIF has children

                cerif_element = root.find('.//{urn:xmlns:org:eurocris:cerif-1.5-1-FRIS}CERIF')
                if cerif_element is not None and len(cerif_element) == 0 or root.find('.//ns2:totalResults',namespaces).text == '0':
                    print(f"Empty page detected at page {page}. Stopping fetch.")
                    break

            except ET.ParseError:
                print(f"Error parsing XML on page {page}. Retrying page.")
                retries += 1
                time.sleep(10)
                continue

            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

            with open(destination_path, 'wt') as f:
                f.write('''<?xml version="1.0" encoding="ISO-8859-1"?>''' + clean_cerif_xml_data(response.text))
                f.close()
            print("\r ...Page %s" % page)
            page += 1
            #time.sleep(5)
            retries = 0
    print(skippedpage)


def fetch_organisations():
    print("\nFETCH ORGANISATIONS...")
    url = "https://frisr4.researchportal.be/ws/OrganisationService"
    headers = {"content-type": "application/xml"}
    body = """
            <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:fris="http://fris.ewi.be/" xmlns:crit="http://fris.ewi.be/criteria">
               <soapenv:Header/>
               <soapenv:Body>
                  <fris:getOrganisations>
                     <crit:organisationCriteria>
                        <crit:window>
                           <crit:pageSize>5000</crit:pageSize>
                           <crit:pageNumber>%s</crit:pageNumber>
                           <orderings>
                                <order>
                                    <id>entity.created</id>
                                    <direction>DESCENDING</direction>
                                </order>
                            </orderings>
                        </crit:window>
                        <crit:dataProviders negated="false">
                        <crit:identifier>KULeuven</crit:identifier>
                        <crit:identifier>Bodemkundige_Dienst_van_Belgie</crit:identifier>
                        <crit:identifier>Thomas_More_Kempen</crit:identifier>
                        <crit:identifier>LSEC</crit:identifier>
                        <crit:identifier>Biogas-E</crit:identifier>
                        <crit:identifier>Thomas_More_Mechelen</crit:identifier>
                        <crit:identifier>WTCB</crit:identifier>
                        <crit:identifier>Plantentuin</crit:identifier>
                        <crit:identifier>Arteveldehogeschool</crit:identifier>
                        <crit:identifier>Centexbel</crit:identifier>
                        <crit:identifier>DSPvalley</crit:identifier>
                        <crit:identifier>Hogeschool_Gent</crit:identifier>
                        <crit:identifier>ILVO</crit:identifier>
                        <crit:identifier>ITG</crit:identifier>
                        <crit:identifier>Proefcentrum_Sierteelt</crit:identifier>
                        <crit:identifier>VIL</crit:identifier>
                        <crit:identifier>FlandersFood</crit:identifier>
                        <crit:identifier>UAntwerpen</crit:identifier>
                        <crit:identifier>Pack4Food</crit:identifier>
                        <crit:identifier>PSGroenteteelt</crit:identifier>
                        <crit:identifier>Departement_Omgeving</crit:identifier>
                        <crit:identifier>KMDA</crit:identifier>
                        <crit:identifier>FlandersBikeValley</crit:identifier>
                        <crit:identifier>VLIZ</crit:identifier>
                        <crit:identifier>SIM</crit:identifier>
                        <crit:identifier>Provincie_Vlaams_Brabant</crit:identifier>
                        <crit:identifier>Departement_MOW</crit:identifier>
                        <crit:identifier>PCAardappelteelt</crit:identifier>
                        <crit:identifier>Katholieke_hogeschool_VIVES_Zuid</crit:identifier>
                        <crit:identifier>VUBrussel</crit:identifier>
                        <crit:identifier>PCHoogstraten</crit:identifier>
                        <crit:identifier>Odisee</crit:identifier>
                        <crit:identifier>BILastechniek</crit:identifier>
                        <crit:identifier>Inagro</crit:identifier>
                        <crit:identifier>KMSKA</crit:identifier>
                        <crit:identifier>SIRRIS</crit:identifier>
                        <crit:identifier>PCFruit</crit:identifier>
                        <crit:identifier>AlamireFoundation</crit:identifier>
                        <crit:identifier>UGent</crit:identifier>
                        <crit:identifier>UCLL</crit:identifier>
                        <crit:identifier>Hogeschool_PXL</crit:identifier>
                        <crit:identifier>UHasselt</crit:identifier>
                        <crit:identifier>HogereZeevaartSchool</crit:identifier>
                        <crit:identifier>PCGroenteteelt</crit:identifier>
                        <crit:identifier>Karel_de_Grote_Hogeschool</crit:identifier>
                        <crit:identifier>INBO</crit:identifier>
                        <crit:identifier>Hogeschool_West-Vlaanderen</crit:identifier>
                        </crit:dataProviders>
                        <crit:external>false</crit:external>
                     </crit:organisationCriteria>
                  </fris:getOrganisations>
               </soapenv:Body>
            </soapenv:Envelope>
            """

    body_new = """
    <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
        <soap:Body>
            <ns1:getOrganisations xmlns:ns1="http://fris.ewi.be/">
                <organisationCriteria xmlns="http://fris.ewi.be/criteria">
                    <window>
                        <pageSize>1000</pageSize>
                        <pageNumber>%s</pageNumber>
                        <orderings>
                            <order>
                                <id>entity.created</id>
                                <direction>DESCENDING</direction>
                            </order>
                        </orderings>
                    </window>
                </organisationCriteria>
            </ns1:getOrganisations>
        </soap:Body>
    </soap:Envelope>
    """
    fetch_from_service(url, headers, body_new, max_pages=50, destination="data_organisations/page%s.xml")


def fetch_persons():
    print("\nFETCH PERSONS...")
    url = "https://frisr4.researchportal.be/ws/PersonService"
    headers = {"content-type": "application/xml"}
    body = """
            <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
            <soap:Body>
                <ns1:getPersons xmlns:ns1="http://fris.ewi.be/">
                    <personCriteria xmlns="http://fris.ewi.be/criteria">
                        <window>
                            <pageSize>5000</pageSize>
                            <pageNumber>%s</pageNumber>
                            <orderings>
                                <order>
                                    <id>entity.created</id>
                                    <direction>DESCENDING</direction>
                                </order>
                            </orderings>
                        </window>
                        <external>false</external>
                    </personCriteria>
                </ns1:getPersons>
            </soap:Body>
        </soap:Envelope>
        """
    fetch_from_service(url, headers, body, max_pages=20, destination="data_persons/page%s.xml")


def fetch_projects():
    print("\nFETCH PROJECTS...")
    url = "https://frisr4.researchportal.be/ws/ProjectService"
    headers = {"content-type": "application/xml"}
    body = """
            <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
                <soap:Body>
                    <ns1:getProjects xmlns:ns1="http://fris.ewi.be/">
                        <projectCriteria xmlns="http://fris.ewi.be/criteria">
                            <window>
                                <pageSize>100</pageSize>
                                <pageNumber>%s</pageNumber>
                                <orderings>
                                    <order>
                                        <id>entity.created</id>
                                        <direction>DESCENDING</direction>
                                    </order>
                                </orderings>
                            </window>
                        </projectCriteria>
                    </ns1:getProjects>
                </soap:Body>
            </soap:Envelope>
            """
    body_fris = """
        <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:fris="http://fris.ewi.be/" xmlns:crit="http://fris.ewi.be/criteria">
        <soapenv:Header/>
            <soapenv:Body>
                <fris:getProjects>
                <crit:projectCriteria>
                    <crit:window>
                        <crit:pageSize>500</crit:pageSize>
                        <crit:pageNumber>%s</crit:pageNumber>
                    </crit:window>
                    <crit:lastModifiedDate>
                        <crit:start>2019-01-01T00:00:00Z</crit:start>
                    </crit:lastModifiedDate>
                </crit:projectCriteria>
            </fris:getProjects>
        </soapenv:Body>
        </soapenv:Envelope>
    """
    fetch_from_service(url, headers, body_fris, max_pages=200, destination='data_projects_2024_5/page%s.xml')
    """
    <crit:start inclusive="false">2019-01-01T00:00:00Z</crit:start>

                        <crit:dataProviders>
                            <crit:identifier>UHasselt</crit:identifier>
                        </crit:dataProviders>
    """

def fetch_journals():
    print("\nFETCH JOURNALS")
    url = "https://frisr4.researchportal.be/ws/JournalServicePublic"
    headers = {"content-type": "application/xml"}
    body = """
            <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
            <soap:Body>
                <ns1:getJournals xmlns:ns1="http://fris.ewi.be/">
                    <criteria xmlns:crit="http://fris.ewi.be/criteria">
                        <crit:window>
                            <crit:pageSize>5000</crit:pageSize>
                            <crit:pageNumber>%s</crit:pageNumber>
                            <crit:orderings>
                                <crit:order>
                                    <crit:id>entity.created</crit:id>
                                    <crit:direction>DESCENDING</crit:direction>
                                </crit:order>
                            </crit:orderings>
                        </crit:window>
                        <crit:dataProviders negated="false">
                            <crit:identifier>KULeuven</crit:identifier>
                            <crit:identifier>Bodemkundige_Dienst_van_Belgie</crit:identifier>
                            <crit:identifier>Thomas_More_Kempen</crit:identifier>
                            <crit:identifier>LSEC</crit:identifier>
                            <crit:identifier>Biogas-E</crit:identifier>
                            <crit:identifier>Thomas_More_Mechelen</crit:identifier>
                            <crit:identifier>WTCB</crit:identifier>
                            <crit:identifier>Plantentuin</crit:identifier>
                            <crit:identifier>Arteveldehogeschool</crit:identifier>
                            <crit:identifier>Centexbel</crit:identifier>
                            <crit:identifier>DSPvalley</crit:identifier>
                            <crit:identifier>Hogeschool_Gent</crit:identifier>
                            <crit:identifier>ILVO</crit:identifier>
                            <crit:identifier>ITG</crit:identifier>
                            <crit:identifier>Proefcentrum_Sierteelt</crit:identifier>
                            <crit:identifier>VIL</crit:identifier>
                            <crit:identifier>FlandersFood</crit:identifier>
                            <crit:identifier>UAntwerpen</crit:identifier>
                            <crit:identifier>Pack4Food</crit:identifier>
                            <crit:identifier>PSGroenteteelt</crit:identifier>
                            <crit:identifier>Departement_Omgeving</crit:identifier>
                            <crit:identifier>KMDA</crit:identifier>
                            <crit:identifier>FlandersBikeValley</crit:identifier>
                            <crit:identifier>VLIZ</crit:identifier>
                            <crit:identifier>SIM</crit:identifier>
                            <crit:identifier>Provincie_Vlaams_Brabant</crit:identifier>
                            <crit:identifier>Departement_MOW</crit:identifier>
                            <crit:identifier>PCAardappelteelt</crit:identifier>
                            <crit:identifier>Katholieke_hogeschool_VIVES_Zuid</crit:identifier>
                            <crit:identifier>VUBrussel</crit:identifier>
                            <crit:identifier>PCHoogstraten</crit:identifier>
                            <crit:identifier>Odisee</crit:identifier>
                            <crit:identifier>BILastechniek</crit:identifier>
                            <crit:identifier>Inagro</crit:identifier>
                            <crit:identifier>KMSKA</crit:identifier>
                            <crit:identifier>SIRRIS</crit:identifier>
                            <crit:identifier>PCFruit</crit:identifier>
                            <crit:identifier>AlamireFoundation</crit:identifier>
                            <crit:identifier>UGent</crit:identifier>
                            <crit:identifier>UCLL</crit:identifier>
                            <crit:identifier>Hogeschool_PXL</crit:identifier>
                            <crit:identifier>UHasselt</crit:identifier>
                            <crit:identifier>HogereZeevaartSchool</crit:identifier>
                            <crit:identifier>PCGroenteteelt</crit:identifier>
                            <crit:identifier>Karel_de_Grote_Hogeschool</crit:identifier>
                            <crit:identifier>INBO</crit:identifier>
                            <crit:identifier>Hogeschool_West-Vlaanderen</crit:identifier>
                        </crit:dataProviders>
                        <crit:external>true</crit:external>
                    </criteria>
                </ns1:getJournals>
            </soap:Body>
        </soap:Envelope>
        """
    fetch_from_service(url, headers, body, max_pages=12, destination="data_journals\\page%s.xml")


def fetch_publications():
    print("\nFETCH PUBLICATIONS...")
    url = "https://frisr4.researchportal.be/ws/ResearchOutputService"
    headers = {"content-type": "application/xml"}
    dataproviders = [#"KULeuven" "UAntwerpen","VUBrussel", "UGent", (unfinished), 
        #Done: "Bodemkundige_Dienst_van_Belgie","Thomas_More_Kempen", Done "LSEC","Biogas-E","Thomas_More_Mechelen","WTCB","Plantentuin","Arteveldehogeschool","Centexbel","DSPvalley","Hogeschool_Gent","ILVO","ITG","Proefcentrum_Sierteelt","VIL","FlandersFood", "Pack4Food","PSGroenteteelt","Departement_Omgeving","KMDA","FlandersBikeValley","VLIZ","SIM","Provincie_Vlaams_Brabant","Departement_MOW","PCAardappelteelt","Katholieke_hogeschool_VIVES_Zuid","PCHoogstraten","Odisee","BILastechniek","Inagro","KMSKA","SIRRIS","PCFruit","AlamireFoundation",
        "UCLL","Hogeschool_PXL","UHasselt","HogereZeevaartSchool","PCGroenteteelt","Karel_de_Grote_Hogeschool","INBO","Hogeschool_West-Vlaanderen"
    ]
    
    for dataprovider in dataproviders:
        print("\nFETCH PUBLICATIONS FOR " + dataprovider)
        body = """
            <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
            <soap:Body>
                <ns1:getResearchOutput xmlns:ns1="http://fris.ewi.be/">
                    <researchOutputCriteria xmlns="http://fris.ewi.be/criteria">
                        <window>
                            <pageSize>100</pageSize>
                            <pageNumber>%s</pageNumber>
                        </window>
                        <dataProviders>
                            <identifier>"""+dataprovider +"""</identifier>
                        </dataProviders>
                        <type>
                            <identifier>Journal Article</identifier>
                        </type>
                    </researchOutputCriteria>
                </ns1:getResearchOutput>
            </soap:Body>
        </soap:Envelope>
            """
        fetch_from_service(url, headers, body, max_pages=1500, destination="data_publications_2024_5\\" + dataprovider + "page%s.xml")

"""

"""
def fetch_funding_code():
    print("\nFETCH FUNDING...")
    url = "https://frisr4.researchportal.be/ws/FundingCodeServiceFRIS"
    headers = {"content-type": "application/xml"}
    body_fris = """
        <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:fris="http://fris.ewi.be/" xmlns:crit="http://fris.ewi.be/criteria">
           <soapenv:Header/>
           <soapenv:Body>
              <fris:getFundingCodes>
                 <criteria>
                    <crit:window>
                       <crit:pageSize>1000</crit:pageSize>
                       <crit:pageNumber>%s</crit:pageNumber>
                    </crit:window>
                 </criteria>
              </fris:getFundingCodes>
           </soapenv:Body>
        </soapenv:Envelope>
    """
    fetch_from_service(url, headers, body_fris, max_pages=1, destination='fundingcode/page%s.xml')

# fetch_organisations()
# fetch_persons()
# fetch_projects()
# fetch_journals()
fetch_publications()
# fetch_funding_code()


