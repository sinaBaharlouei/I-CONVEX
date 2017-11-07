from DatasetGenerator import similarPairsGenerator, dissimilarPairsGenerator
from random import randint
import csv

# generate exons
# generate transcripts from exons
number_of_reads = 50000
number_of_clusters = 5000
number_of_exons = 400
exome_length = 20
number_of_exons_generating_transcript = 20

exons = dissimilarPairsGenerator.generateRandomExons(exome_length, number_of_exons)
transcripts = dissimilarPairsGenerator.generate_transcripts(exons, number_of_clusters, number_of_exons_generating_transcript)

with open('dataset' + str(number_of_reads) + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Sequence', 'ClusterIndex'])

    for i in range(number_of_reads):
        chosen_transcript_index = randint(0, number_of_clusters-1)
        chosen_transcript = transcripts[chosen_transcript_index]

        shifted = similarPairsGenerator.shiftString(chosen_transcript, 5)
        noisy = similarPairsGenerator.noise(shifted, 0.02, 0.02)

        if len(noisy) > len(chosen_transcript):
            noisy = noisy[:len(chosen_transcript)]  # cut noisy version of transcript

        elif len(chosen_transcript) > len(noisy):
            res = similarPairsGenerator.generateRandomRNA(len(chosen_transcript) - len(noisy))
            noisy += res

        writer.writerow([noisy, chosen_transcript_index])