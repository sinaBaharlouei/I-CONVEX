from Bio import SeqIO
import random
import sys
import csv


def read_fasta_file(fileName):
    return list(SeqIO.parse(fileName, 'fasta'))


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


def split_file(filename, chunk_numbers):
    fasta_file = read_fasta_file(filename)
    number_of_reads = len(fasta_file)
    print(number_of_reads)
    if chunk_numbers <= 0:
        if number_of_reads % 50000 == 0:
            chunk_numbers = number_of_reads // 50000
        else:
            chunk_numbers = number_of_reads // 50000 + 1

    batch_size = number_of_reads // chunk_numbers

    for i in range(chunk_numbers - 1):
        start = i * batch_size
        end = start + batch_size

        sequences = []
        for j in range(start, end):
            sequences.append(fasta_file[j])

        SeqIO.write(sequences, "chunk" + str(i + 1) + ".fasta", "fasta")

    # last file
    start = (chunk_numbers - 1) * batch_size
    end = number_of_reads
    sequences = []
    for j in range(start, end):
        sequences.append(fasta_file[j])

    SeqIO.write(sequences, "chunk" + str(chunk_numbers) + ".fasta", "fasta")
    return chunk_numbers


number_of_chunks = -1
number_of_parameters = len(sys.argv)

if number_of_parameters > 1:
    number_of_chunks = sys.argv[1]


if number_of_parameters > 4:
    r = int(sys.argv[2])
    b = int(sys.argv[3])
    k = int(sys.argv[4])
else:
    r = 1
    b = 10
    k = 15

print("r = ", r)
print("b = ", b)
print("k = ", k)
hash_functions = createHashFunctions(r * b, 1000)
number_of_chunks = split_file('reads.fasta', number_of_chunks)

with open('hash_functions.csv', 'wb') as f:
    w = csv.writer(f, delimiter=',')
    w.writerow([r, b, k, number_of_chunks])
    for item in hash_functions:
        w.writerow([item[0], item[1]])

"""Create Bash File"""
S = ""
run_counter = (number_of_chunks-1) // 20 + 1
for i in range(run_counter):
    S += "python MPMH.py " + str(i+1) + "\n"
S += "python MultiProcessLSH.py\n"
S += "chmod 777 collector.sh\n"
S += "./collector.sh\n"
S += "python FastaPreparation.py\n"
S += "python ValidatePairs.py"
f = open("commands.sh", "w+")
f.write(S)
f.close()

