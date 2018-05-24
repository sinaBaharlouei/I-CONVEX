from Bio import SeqIO
from DataOperations.FastaIO import read_fasta_file
import random
import sys
import csv


def createHashFunctions(n, p):
    """
    :param n: number of hash functions
    :param p: length of permutation
    :return: Each hash function could be represented by pair (a,b) meaning h(c) = (ac + b) % p
    """
    if p > 1000:
        p = 1000

    hash_function_pairs = set()
    while len(hash_function_pairs) < n:
        hash_function_pairs.add((random.randint(1, p - 1), random.randint(1, p - 1)))

    return list(hash_function_pairs)
def split_file(filename, number_of_chunks):

    fasta_file = read_fasta_file(filename)
    number_of_reads = len(fasta_file)

    if number_of_chunks <= 0:
        number_of_chunks = number_of_reads // 100000

    batch_size = number_of_reads // number_of_chunks

    for i in range(number_of_chunks - 1):
        start = i * batch_size
        end = start + batch_size

        sequences = []
        for j in range(start, end):
            sequences.append(fasta_file[j])

        SeqIO.write(sequences, "chunk" + str(i+1) + ".fasta", "fasta")

    # last file
    start = (number_of_chunks - 1) * batch_size
    end = number_of_reads
    sequences = []
    for j in range(start, end):
        sequences.append(fasta_file[j])

    SeqIO.write(sequences, "chunk" + str(number_of_chunks) + ".fasta", "fasta")


number_of_chunks = -1
number_of_parameters = len(sys.argv)

if number_of_parameters > 1:
    number_of_chunks = sys.argv[1]

split_file('reads1M.fasta', number_of_chunks)

if number_of_parameters > 3:
    r = int(sys.argv[2])
    b = int(sys.argv[3])

else:
    r = 1
    b = 10

hash_functions = createHashFunctions(r*b, 1000)
with open('hash_functions.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.writer(f, delimiter=',')
    w.writerow([r, b, number_of_chunks])
    for item in hash_functions:
        w.writerow([item[0], item[1]])