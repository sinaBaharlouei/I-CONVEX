import random
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def noise(read, Pi, Pd):
    """
    :param read: noiseless input
    :param  Pi: the probability of inserting random letter at each index
    :param  Pd: the probability of deleting the letter in certain position(insertion and deletion)
    :return:
    """
    letterList = ['A', 'C', 'G', 'T']

    b = bytearray(read, 'utf8')
    l = len(b)
    j = 0
    for i in range(l):
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
number_of_transcripts = len(transcripts)
number_of_generated_reads = 1000000
ground_truth_cluster_id_list = []
records = []
for i in range(number_of_generated_reads):
    chosen_transcript_index = random.randint(0, number_of_transcripts - 1)
    chosen_transcript = transcripts[chosen_transcript_index]
    new_read = noise(shiftString(chosen_transcript, 5), 0.02, 0.02)
    print(chosen_transcript)
    print(new_read)
    print(chosen_transcript_index)
    record = SeqRecord(Seq(new_read), 'm_' + str(i+1000), '', '')
    records.append(record)
    ground_truth_cluster_id_list.append(chosen_transcript_index)

    if i > 100:
        break


SeqIO.write(records, "reads1M.fasta", "fasta")
