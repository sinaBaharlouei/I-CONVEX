import numpy as np
import math
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import csv
import random


def get_abundance_vector():
    rho = np.random.randn(1, number_of_transcripts)

    print(rho)
    exp_rho = []

    print(len(rho[0]))
    sum_of_rhos = 0

    for i in range(len(rho[0])):
        current_value = math.exp(rho[0][i])
        exp_rho.append(current_value)
        sum_of_rhos += current_value

    for i in range(len(exp_rho)):
        exp_rho[i] /= sum_of_rhos

    print(exp_rho)

    abundance_vector = []
    for i in range(len(exp_rho)):
        abundance_vector.append(np.random.poisson(exp_rho[i] * average_num_reads))

    print(max(abundance_vector))
    print(abundance_vector)
    print(sum(abundance_vector))
    return abundance_vector


def noise(read, Pi, Pd):
    """
    :param read: noiseless input
    :param  Pi: the probability of inserting random letter at each index
    :param  Pd: the probability of deleting the letter in certain position(insertion and deletion)
    :return:
    """
    letterList = ['A', 'C', 'G', 'T']

    b = bytearray(read, 'utf8')
    j = 0
    for t in range(len(b)):
        is_inserted = is_deleted = False
        p = random.random()
        if p < Pd:
            del b[j]
            is_deleted = True

        p = random.random()
        if p < Pi:
            b = b[0:j] + bytearray(letterList[random.randint(0, 3)], 'utf8') + b[j:]
            is_inserted = True

        if not is_inserted and not is_deleted:
            j += 1
        elif is_inserted and not is_deleted:
            j += 2
        elif is_inserted and is_deleted:
            j += 1

    return b.decode(encoding='utf-8')


def shiftString(read, shift_index_range):
    """
    read: input RNA string
    """
    letterList = ['A', 'C', 'G', 'T']
    beginningShiftIndex = random.randint(-shift_index_range, shift_index_range)
    endShiftIndex = random.randint(-shift_index_range, shift_index_range)

    if beginningShiftIndex < 0:  # insert some letters to the beginning of string
        str1 = ''
        for i in range(-beginningShiftIndex):
            str1 += letterList[random.randint(0, 3)]
        read = str1 + read

    else:
        read = read[beginningShiftIndex:]

    if endShiftIndex < 0:  # add some letters to the end
        str2 = ""
        for i in range(-endShiftIndex):
            str2 += letterList[random.randint(0, 3)]
        read += str2

    elif endShiftIndex > 0:  # remove some letters from the end
        read = read[:-endShiftIndex]
    return read


file = open("GTcenters.txt", "r")
reads = file.read()
transcripts = reads.split('\n')
transcripts.pop(5000)
average_num_reads = 1000000
number_of_transcripts = 5000

rho_vec = get_abundance_vector()
ground_truth_cluster_id_list = []
records = []
n = 0

indices = []


for j in range(len(rho_vec)):

    chosen_transcript = transcripts[j]
    for k in range(rho_vec[j]):
        new_read = noise(shiftString(chosen_transcript, 3), 0.02, 0.02)
        record = SeqRecord(Seq(new_read), 'm_' + str(n + 1000), '', '')
        records.append(record)
        ground_truth_cluster_id_list.append(j)
        n += 1


for item in range(n):
    indices.append(item)

print(indices[0])
print(indices[100])
random.shuffle(indices)
print(indices)

final_records = []
t = 0
for item in indices:
    new_rec = records[item]
    new_rec.id = 'm_' + str(t)
    final_records.append(new_rec)
    t += 1
print(final_records)

SeqIO.write(final_records, "unbalanced1M.fasta", "fasta")
with open('unbalanced1M.csv', 'w', newline='') as f:  # Just use 'w' mode in 3.x
    w = csv.writer(f, delimiter=',')
    for item in indices:
        w.writerow([ground_truth_cluster_id_list[item]])