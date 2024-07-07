import csv

# OUTPUT OF THIS CODE IS common_terms.csv
# There are 51571 terms in terimler.org files we received via email from Bülent Hoca
# There are 23408 terms in our final_one2one file
# We have 23388 matching terms between final_one2one.csv and terimlerExcelExtracted.csv (terimler.org files)
file1 = open('terms/final/final_one2one.csv', encoding='utf-8')
final_reader = csv.reader(file1)

file2 = open('terms/terimlerExcelExtracted.csv', encoding='utf-8')
terimler_reader = csv.reader(file2)

terimler_set = set()

for row in terimler_reader:
    term = row[0]
    term.lower()
    terimler_set.add(term)

file2.close()

output_file = open('common_terms.csv', 'w', newline='', encoding='utf-8')
output_writer = csv.writer(output_file)

for row in final_reader:
    finalTerm = row[1]
    finalTerm.lower()
    if finalTerm in terimler_set:
        output_writer.writerow([finalTerm])


file1.close()
output_file.close()
