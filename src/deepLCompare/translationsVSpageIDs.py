import csv
import requests

file1 = open('translationIDpageID.csv', encoding='utf-8')
translationID_reader = csv.reader(file1)

output_file = open('translationID2translation.csv', 'w', newline='', encoding='utf-8')
output_writer = csv.writer(output_file)

URL = "https://en.wikipedia.org/w/api.php"
PARAMS = {
    "action": "query",
    "format": "json",
    "list": "contenttranslationcorpora",
    "striphtml": "true",
    "translationid": ""
}

i = 0
for row in translationID_reader:  # 0-255th rows are marked TRUE which means these pages are scientific
    if i <= 255:
        # Find translation data
        print(i)
        PARAMS["translationid"] = row[0]
        RESPONSE = requests.get(url=URL, params=PARAMS)
        responseDict = RESPONSE.json()
        translation = responseDict["query"]["contenttranslationcorpora"]["sections"]

        # row[0] = translationID, row[4] = page title EN, row[2] = wikilinkEN
        output_writer.writerow([row[0], row[4], row[2], translation])
        i += 1

output_file.close()
file1.close()



