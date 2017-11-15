import csv
import random
from Bio import SeqIO


def generateRandomRNA(length):
    """
    length: Length of the generated RNA
    """
    letterList = ['A', 'C', 'G', 'T']
    res = ""
    for i in range(length):
        res += letterList[random.randint(0, 3)]

    return res


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


def generateSimilarPairs(filename, read_length, read_numbers):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Seq1', 'Seq2'])
        for i in range(read_numbers):
            p1 = generateRandomRNA(read_length)
            p2 = noise(shiftString(p1, 5), 0.02, 0.02)

            if len(p2) > len(p1):
                p2 = p2[:len(p1)]  # cut p2

            elif len(p1) > len(p2):
                res = generateRandomRNA(len(p1) - len(p2))
                p2 += res
            writer.writerow([p1, p2])
