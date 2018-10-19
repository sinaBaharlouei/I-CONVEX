from Bio import SeqIO
import csv


def read_fasta_file(fileName):
    return list(SeqIO.parse(fileName, 'fasta'))


def trim(sequence):
    w1 = len(sequence)
    w = 400
    if w1 < w:
        for k in range(w - w1):
            sequence += 'A'
    else:
        sequence = sequence[:400]

    return sequence


reads = read_fasta_file('reads.fasta')

for i in range(len(reads)):
    reads[i].seq = trim(reads[i].seq)

final_list = []
final_ids = []

SeqIO.write(reads, 'trimmed.fasta', 'fasta')