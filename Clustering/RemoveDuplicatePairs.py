import csv

pair_set = set()
with open('merged.csv', 'r') as csvfile:
    pairs = csv.reader(csvfile, delimiter=',')
    for item in pairs:
        pair_set.add((item[0], item[1]))

print(len(pair_set))

with open('G.csv', 'wb') as f:
    w = csv.writer(f, delimiter=',')
    for item in pair_set:
        w.writerow([item[0], item[1]])
