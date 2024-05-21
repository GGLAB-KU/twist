import csv

file1 = open('terimler_org_all.csv', encoding='utf-8')
terimler_reader = csv.reader(file1)
file2 = open('IATE_export.csv', encoding='utf-8')
iate_reader = csv.reader(file2,  delimiter='|')



with open("terimler_IATE_comparison.csv", 'w', newline='', encoding='utf-8') as output:
    writer = csv.writer(output)
    writer.writerow(["t-id", "t-name", "t-slug", "t-number_of_definitions", "t-scope", "t-content", "t-number_of_synonyms", "t-synonyms", "t-number_of_en", "t-en", "exist in IATE", "i-id", "i-domains", "i-term", "i-type" ])

 #   E_ID | E_DOMAINS | L_CODE | T_TERM | T_TYPE | T_RELIABILITY | T_EVALUATION | E_PRIMARITY | E_LIFECYCLE | E_ORIGINS | T_INSTITUTION | T_MODIFICATION_TIMESTAMP
# i 0-1, 3-4,

#    id, name, slug, number_of_definitions, scope, content, number_of_synonyms, synonyms, number_of_en,en
# t 0-8, 12

    for trow in terimler_reader:
        found = False
        term = trow[12]
        if term.find(';') != -1:
            term.split(sep="; ")
            for t in term:
                print(f"t: {t}")
                for irow in iate_reader:
                    if t == irow[1]:
                        writer.writerow(
                            [trow[0], trow[1], trow[2], trow[3], trow[4], trow[5], trow[6], trow[7], trow[8], trow[12],
                             'True', irow[0], irow[1], irow[3], irow[4]])
                        found = True
                        continue

        else:
            for irow in iate_reader:
                print(f"term2: {term}")
                if term == irow[3]:
                    writer.writerow(
                        [trow[0], trow[1], trow[2], trow[3], trow[4], trow[5], trow[6], trow[7], trow[8], trow[12],
                         'True', irow[0], irow[1], irow[3], irow[4]])
                    found = True
                    continue
        if not found:
            writer.writerow([trow[0], trow[1], trow[2], trow[3], trow[4], trow[5], trow[6], trow[7], trow[8], trow[12], 'False'])
