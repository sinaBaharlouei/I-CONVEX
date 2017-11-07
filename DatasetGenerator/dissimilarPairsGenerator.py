import random
from random import shuffle
import csv


def generateRandomExons(exomeLength, multiplicity):
    letters_list = ['A', 'C', 'G', 'T']
    exons = []
    for i in range(multiplicity):
        current_exome = ""
        for j in range(exomeLength):
            current_exome += letters_list[random.randint(0, 3)]
        exons.append(current_exome)
    return exons


def generate_transcripts(exons, multiplicity, number_of_exons_in_transcript):
    n = len(exons)
    transcripts = []
    for counter in range(multiplicity):

        base_sequence = []
        for i in range(n):
            base_sequence.append(i)

        shuffle(base_sequence)
        chosen_exons = sorted(base_sequence[:number_of_exons_in_transcript])

        seq1 = ""
        for index in chosen_exons:
            seq1 += exons[index]

        transcripts.append(seq1)

    return transcripts


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

        pair.append(seq1)

    return pair


def generateFile(fileName, multiplicity, number_of_exons, exons_length, number_of_exons_in_read):
    with open(fileName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Seq1', 'Seq2'])
        exons = generateRandomExons(exons_length, number_of_exons)
        for i in range(multiplicity):
            p1, p2 = generateDissimilarPairs(exons, number_of_exons_in_read)
            writer.writerow([p1, p2])
