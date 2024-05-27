import requests
import csv
import re


def glossary_csv(csv_file_name, param_titles, page_id):

    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "prop": "revisions",
        "titles": param_titles,
        "rvslots": "*",
        "rvprop": "content",
        "format": "json",
    }

    with open(csv_file_name, 'w', newline='', encoding='utf-8') as output:

        writer = csv.writer(output)
        RESPONSE = requests.get(url=URL, params=PARAMS)
        responseDict = RESPONSE.json()
        content = responseDict["query"]["pages"][page_id]["revisions"][0]["slots"]["main"]["*"]
        list_of_matches = re.findall(r'term\|\[\[.+\]\]', content)

        for noisy in list_of_matches:
            match = noisy[7:-2]
            multiple_words = match.find('|')

            # if there are 2 words with same meaning
            if multiple_words != -1:
                # depart words
                word1 = match[0:multiple_words]
                word2 = match[multiple_words+1:]
                # print(f"word1: {word1}, word2: {word2}")
                # output: Sublimation (phase transition), sublimation

                # delete in paranthesis explanations
                paranthesis = word1.find('(')
                if paranthesis != -1:
                    word1 = word1[0: paranthesis]

                paranthesis = word2.find('(')
                if paranthesis != -1:
                    word2 = word2[0: paranthesis]

                # write to csv
                writer.writerow([word1, word2])

            # if there is only one word
            else:
                writer.writerow([match])


glossary_csv("glossary_of_physics.csv", "Glossary_of_physics", "36626070")
glossary_csv("glossary_of_computer_science.csv", "Glossary_of_computer_science", "57143357")
glossary_csv("glossary_of_areas_of_mathematics.csv", "Glossary_of_areas_of_mathematics", "34189212")
