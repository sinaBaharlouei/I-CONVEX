import random
from queue import Queue
import community

from DataOperations import FastaIO, KGrams, graphOperations
import networkx as nx
import timeit
import matplotlib.pyplot as plt


def LSH(signature_matrix, reads, b, r):
    pass


def getMinHashFunctions(reads, k, r, b, number_of_permutations):
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
        k_gram_dictionary[read.id] = KGrams.find_k_grams(read.seq, k)

    now = timeit.default_timer()
    print("k-mers representation is ready.", now - start)
    """
    
    for read_id1 in k_gram_dictionary:
        for read_id2 in k_gram_dictionary:
            if read_id1 != read_id2:
                print(len(set(k_gram_dictionary[read_id1]).intersection(set(k_gram_dictionary[read_id2]))) / len(set(k_gram_dictionary[read_id1]).union(set(k_gram_dictionary[read_id2]))))
    """

    # get hash functions
    hash_functions = createHashFunctions(n, p)

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
            for item in k_gram_dictionary[key]:  # each shingle in string
                current_number_hash_value = (a * item + b) % p
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
    b = n // r

    # Create graph
    G = nx.Graph()
    for read_id in read_id_list:
        G.add_node(read_id)

    """
    for key in band_dictionary:
        if len(band_dictionary[key]) > 1:
            for i in range(1, len(band_dictionary[key])):
                for j in range(i):
                    G.add_edge(band_dictionary[key][j], band_dictionary[key][i])
    FastaIO.write_graph_to_csv('WGraphK' + str(k) + 'R' + str(r) + 'B' + str(b) + '.csv', G)
    """
    for key in band_dictionary:
        L = len(band_dictionary[key])
        t = 1
        while L > t and t < number_of_permutations:
            for i in range(L-t):
                G.add_edge(band_dictionary[key][i], band_dictionary[key][i+t])
            t += 1
    FastaIO.write_graph_to_csv('MinGraphK' + str(k) + 'R' + str(r) + 'B' + str(b) + 'P' + str(number_of_permutations) + '.csv', G)

    now4 = timeit.default_timer()
    print("Graph is created.", now4 - now3)

    print(len(G.edges()))

    return LSH_dict


def createRepresentationMatrix(k, row_numbers, columns):
    # get random k-grams representing the rows of similarity matrix
    # k-grams are hashed to 32bit integers for more convenience.
    all_6_grams = KGrams.generate_all_k_grams(k, ['A', 'C', 'G', 'T'])
    random_grams = KGrams.get_random_k_grams(all_6_grams, row_numbers)

    random_grams_counter = {}

    for random_gram in random_grams:
        # print(random_gram)
        random_grams_counter[random_gram] = 0

    for random_gram in random_grams:
        for key1 in columns:
            if random_gram in columns[key1]:
                random_grams_counter[random_gram] += 1

                # print(random_grams_counter)


def bandHashFunction(integer_list):
    hash_value = 0
    index = 1
    for item in integer_list:
        hash_value += (index * index) * (item + 1) * (item + 2) + (item + 3)
        index += 1
    return hash_value


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


dataset = FastaIO.read_fasta_file('../files/reads400.fasta')

getMinHashFunctions(dataset, 15, 1, 10, 10)

"""
k = 8
k_gram_dictionary = {}
n = 1

for record in dataset:
    k_gram_dictionary[record.id] = KGrams.find_k_grams(record.seq, k)
    if n >= 100:
        break
    n += 1

createRepresentationMatrix(k, 29, k_gram_dictionary)
"""
