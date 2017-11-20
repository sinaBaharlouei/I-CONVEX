from DatasetGenerator.dissimilarPairsGenerator import generateFile
from DatasetGenerator.similarPairsGenerator import generateSimilarPairs
import random
import csv


generateFile("dissimilar500.csv", 500, 400, 20, 20)
generateSimilarPairs("similar500.csv", 400, 500)

with open('similar500.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    similar_pairs = list(reader)
    similar_pairs.pop(0)
    similar_pairs = [x + [1] for x in similar_pairs]

with open('dissimilar500.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    dissimilar_pairs = list(reader)
    dissimilar_pairs.pop(0)
    dissimilar_pairs = [x + [0] for x in dissimilar_pairs]

pairs = similar_pairs + dissimilar_pairs
random.shuffle(pairs)

with open("pair_dataset_1000.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Seq1', 'Seq2', 'Similarity'])
    for item in pairs:
        writer.writerow([item[0], item[1], item[2]])


