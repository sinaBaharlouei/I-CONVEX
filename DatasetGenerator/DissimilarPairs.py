import random
from random import shuffle


def generateRandomExons(exomLength, multiplicity):
    letters_list = ['A', 'C', 'G', 'T']
    exons = []
    for i in range(multiplicity):
        current_exome = ""
        for j in range(exomLength):
            current_exome += letters_list[random.randint(0, 3)]
        exons.append(current_exome)
    return exons


def generateDissimilarPairs(exons, multiplicity):
    n = len(exons)
    pair = []
    for counter in range(2):

        base_sequence = []
        for i in range(n):
            base_sequence.append(i)

        shuffle(base_sequence)
        chosen_exons = sorted(base_sequence[:multiplicity])

        seq1 = ""
        for index in chosen_exons:
            seq1 += exons[index]

        print(chosen_exons)
        print(seq1)
        pair.append(seq1)

    return pair


generateDissimilarPairs(generateRandomExons(20, 100), 10)
