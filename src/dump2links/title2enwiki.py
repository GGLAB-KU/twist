import wikipediaapi
import csv
import time

'''
wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'tr')
page_tr = wiki_wiki.page('Türkiye_ekonomik_krizi_(2018-günümüz)')
page_en = page_tr.langlinks['en']
title_en = page_en.title
'''

file = open('links2titles.csv', encoding="utf-8")
csvreader = csv.reader(file)

with open("tr2en.csv", 'w', newline='', encoding='utf-8') as output:
    writer = csv.writer(output)
    writer.writerow(["translationid", "page link (tr)", "page title (tr)", "page link (en)", "page title (en)"])

    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'tr')
    start_time = time.time()
    i = 1
    for row in csvreader:
        title_tr = row[2]
        if 'en' in wiki_wiki.page(title_tr).langlinks:
            title_en = wiki_wiki.page(title_tr).langlinks['en'].title
            print(f"{time.time() - start_time:.2f} {i} {title_tr}: {title_en}")
            writer.writerow([row[0], row[1], row[2], title_en])
        else:
            print(f"{time.time() - start_time:.2f} {i} {title_tr}: {title_en}")
            writer.writerow([row[0], row[1], row[2], "No match found."])
        i += 1


