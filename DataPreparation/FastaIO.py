from Bio import SeqIO
import random


def read_fasta_file(fileName):
    return list(SeqIO.parse(fileName, 'fasta'))


def read_fasta_file_as_dict(fileName):
    return SeqIO.to_dict(SeqIO.parse(fileName, 'fasta'))
