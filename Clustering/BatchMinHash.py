import csv
import timeit
import sys
from Bio import SeqIO


def read_fasta_file(fileName):
    return list(SeqIO.parse(fileName, 'fasta'))


def find_k_grams(record, k):
    k_gram_array = []
    length = len(record)
    for i in range(length - k):
        k_gram_array.append(hash(str(record[i:i + k])) & 0xffffffff)
        # k_gram_array.append(string2numeric_hash(str(record[i:i + k])))

    # print(k_gram_array)
    return k_gram_array


def getMinHashFunctions(reads, k, r, b):
    print(k)
    """Return signature matrix
    reads: list of reads
    k: length of each one of k-mers
    n: number of hash functions
    b: number of bands
    n = b * m
    """
    n = b * r
    start = timeit.default_timer()

    # first integer number greater than all 32 bit integers
    p = 4294967311

    # get random k-grams representing the rows of similarity matrix
    # k-grams are hashed to 32bit integers for more convenience.
    # all_k_grams = KGrams.generate_all_k_grams(k, ['A', 'C', 'G', 'T'])
    # random_grams = KGrams.get_random_k_grams(all_k_grams, r)

    read_id_list = []

    # create k-mer dictionary for reads:
    k_gram_dictionary = {}
    for read in reads:
        read_id_list.append(read.id)
        k_gram_dictionary[read.id] = find_k_grams(read.seq, k)

    now = timeit.default_timer()
    print("k-mers representation is ready.", now - start)

    # Create Signature dictionary: the key of each element would be the pair (S, i):
    # S denotes the molecular string(read), i denotes the i-th hash function.
    # The value of the element with key(S, i) would be h_i(S)
    signature_dictionary = {}

    for i in range(n):
        signature_dictionary[i] = {}

    """
    # First Sig(S, i) must be initialized to +inf
    for key in k_gram_dictionary:
        for hash_function_index in range(n):
            signature_dictionary[hash_function_index][key] = 9999999999
    """

    # Algorithm Description:
    """
    For each chosen k_gram(rows of matrix):
        1) compute h1(row), h2(row), ...
        2) for each string:
            if k_gram exists in string:
                sig(s, i) = min(sig(s,i), h_i(row))
    """
    hash_index = -1
    for a, b in hash_functions:

        hash_index += 1
        # for each hash function we compute h_i(s) for all strings
        for key in k_gram_dictionary:  # each string
            minHash = 9999999999
            for val in k_gram_dictionary[key]:  # each shingle in string
                current_number_hash_value = (a * val + b) % p
                if current_number_hash_value < minHash:
                    minHash = current_number_hash_value

            signature_dictionary[hash_index][key] = minHash

    now2 = timeit.default_timer()
    print("minHash signature matrix is ready.", now2 - now)

    LSH_dict = {}
    for read_id in read_id_list:
        LSH_dict[read_id] = []

    band_dictionary = {}

    for read_id in read_id_list:

        integer_list = []  # it should have m elements
        hash_function_counter = 0
        band_counter = 0

        for hash_function_index in signature_dictionary:

            integer_list.append(signature_dictionary[hash_function_index][read_id])
            hash_function_counter += 1

            if hash_function_counter == r:
                hashedValue = bandHashFunction(integer_list)
                LSH_dict[read_id].append(hashedValue)

                # add read to the bucket corresponding the band
                band_counter += 1
                band_dictionary.setdefault((band_counter, hashedValue), []).append(read_id)
                hash_function_counter = 0
                integer_list = []

    now3 = timeit.default_timer()
    print("LSH array is ready.", now3 - now2)
    return LSH_dict


def bandHashFunction(integer_list):
    hash_value = 0
    index = 1
    for val in integer_list:
        hash_value += (index * index) * (val + 1) * (val + 2) + (val + 3)
        index += 1
    return hash_value


number_of_parameters = len(sys.argv)

if number_of_parameters > 1:
    file_index = sys.argv[1]
    filename = "chunk" + str(file_index) + ".fasta"
    print(filename)

else:
    print("File name is not assigned.")
    file_index = 1
    filename = "chunk" + str(file_index) + ".fasta"

hash_functions = []
with open('hash_functions.csv', 'r') as csvfile:
    pairs = list(csv.reader(csvfile, delimiter=','))
    rows = int(pairs[0][0])
    bands = int(pairs[0][1])
    k_parameter = int(pairs[0][2])
    pairs.pop(0)
    for pair in pairs:
        hash_functions.append([int(pair[0]), int(pair[1])])

    dataset = read_fasta_file(filename)

    final_lsh_dict = getMinHashFunctions(dataset, k_parameter, rows, bands)
    with open('LSH_' + str(file_index) + '.csv', 'wb') as f:  # Just use 'w' mode in 3.x
        w = csv.writer(f, delimiter=',')
        for item in final_lsh_dict:
            row = [item] + final_lsh_dict[item]
            w.writerow(row)
