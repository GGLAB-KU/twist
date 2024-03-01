import wikipediaapi
import requests
import json
from urllib.parse import quote
import csv
import gzip
import re
import logging
import time

# Create log file
logging.basicConfig(level=logging.INFO, filename="dump2links.log", filemode="w")

# Opening JSON file
path = 'cx-corpora.en2tr.text.json'
dump = open(path, encoding='utf-8')

# returns JSON object as a dictionary
dumpdict = json.load(dump)
dump.close()
logging.info("JSON object converted to dictionary")

# request info
URL = "https://tr.wikipedia.org/w/api.php"
PARAMS = {
    "srsearch": "",
    "action": "query",
    "format": "json",
    "list": "search",
    "srwhat": "text"
}

# open csv file (testing purposes)
with open("dump2links.csv", 'w', newline='') as output:
    writer = csv.writer(output)
    writer.writerow(["translationid", "page link"])

    # Iterating through the json list
    currentid = "0"
    articlefound = False

    start_time = time.time()
    translation_index = 0
    search_counter = 0
    for i in dumpdict:
        search_counter += 1
        translationid = i['id'].split('/')[0]
        if (translationid == currentid) & articlefound:
            print(f"{time.time() - start_time:.2f}  Skipped i: {search_counter} with translationid: {currentid}")
            continue
        else:
            if translationid != currentid:
                translation_index += 1
                if not articlefound:
                    writer.writerow([currentid, "No match found."])

            articlefound = False
            currentid = translationid
            txt_search = i['target']['content']  # extract the content to search for it
            print(f"{time.time() - start_time:.2f}  Searching i: {search_counter} with translationid: {currentid}")
            if len(txt_search) > 40:
                lst_search = txt_search.split()
                txt_search = ""
                x = 0
                while x <= 8 and x < len(lst_search):
                    txt_search += lst_search[x] + " "
                    x += 1
                txt_search = txt_search[:-1]

                PARAMS["srsearch"] = txt_search
                RESPONSE = requests.get(url=URL, params=PARAMS)
                responseDict = RESPONSE.json()

                results = len(responseDict['query']['search'])
                if results == 1:  # search returns specific result
                    pageid = responseDict['query']["search"][0]["pageid"]
                    logging.info(f"Match found for {translationid} "
                                 f"\nText: {txt_search[:200]} "
                                 f"\nLink: https://tr.wikipedia.org/?curid={pageid}")
                    writer.writerow([translationid, f"https://tr.wikipedia.org/?curid={pageid}"])
                    print(f"{time.time() - start_time:.2f}  FOUND i: {search_counter} with translationid: {currentid}")
                    articlefound = True
                elif results >= 1:  # search returns more than one result
                    logging.error(f"More than 1 result found for {translationid}: {txt_search[:200]}")
                    continue
                else:  # search returns nothing
                    logging.error(
                        f"No results found for translation id {translationid}: {txt_search} \nEncoded version: {quote(txt_search)}")
                    continue
            else:
                logging.error(f"Text {txt_search} is too short to be searched.")

        # if translation_index == 300:
        #    break

# Closing file and session
# dump.close()






