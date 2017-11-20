import csv


with open('../dataset100.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    reads = list(reader)

count1 = count2 = 0
with open("pairs_100.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Seq1', 'Seq2', 'Similarity'])

    for i in range(1, len(reads)):
        for j in range(i+1, len(reads)):
            if reads[i][1] != reads[j][1]:
                count1 += 1
                writer.writerow([reads[i][0], reads[j][0], 0])
            else:
                count2 += 1
                writer.writerow([reads[i][0], reads[j][0], 1])
