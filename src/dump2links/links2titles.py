import csv
import requests
import time

file = open('dump2links.csv')
csvreader = csv.reader(file)

URL = "https://tr.wikipedia.org/w/api.php"
PARAMS = {
    "action": "query",
    "prop": "info",
    "pageids": "",
    "inprop": "url",
    "format": "json",
}


with open("links2titles.wrongfile.csv", 'w', newline='', encoding='utf-8') as output:
    writer = csv.writer(output)
    writer.writerow(["translationid", "page link (tr)", "page title (tr)", "page link (en)", "page title (en)"])

    start_time = time.time()
    i = 1
    for row in csvreader:
        trpageid = row[1][32:]
        PARAMS["pageids"] = trpageid
        RESPONSE = requests.get(url=URL, params=PARAMS)
        responseDict = RESPONSE.json()
        title = responseDict["query"]["pages"][trpageid]["title"]
        print(f"{time.time() - start_time:.2f} {i} {trpageid}: {title}")
        writer.writerow([row[0], row[1], title])