import random
from DataPreparation import FastaIO, KGrams


def LSH(signature_matrix, reads, b, r):
    pass


def getMinHashFunctions(reads, k, r, n):
    """Return signature matrix
    reads: list of reads
    k: length of each one of k-mers
    n: number of hash functions
    r: number of rows
    """

    # get random k-grams representing the rows of similarity matrix
    # k-grams are hashed to 32bit integers for more convenience.
    all_6_grams = KGrams.generate_all_k_grams(k, ['A', 'C', 'G', 'T'])
    random_grams = KGrams.get_random_k_grams(all_6_grams, r)

    read_id_list = []

    # create k-mer dictionary for reads:
    k_gram_dictionary = {}
    t = 1
    for read in reads:
        read_id_list.append(read.id)
        k_gram_dictionary[read.id] = KGrams.find_k_grams(read.seq, k)
        if t >= 100:
            break
        t += 1

    # get hash functions
    hash_functions = createHashFunctions(n, r)

    # Create Signature dictionary: the key of each element would be the pair (S, i):
    # S denotes the molecular string(read), i denotes the i-th hash function.
    # The value of the element with key(S, i) would be h_i(S)
    signature_dictionary = {}
    for i in range(n):
        signature_dictionary[i] = {}

    # First Sig(S, i) must be initialized to +inf
    for key in k_gram_dictionary:
        for hash_function_index in range(n):
            signature_dictionary[hash_function_index][key] = 100

    # Algorithm Description:
    """
    For each chosen k_gram(rows of matrix):
        1) compute h1(row), h2(row), ...
        2) for each string:
            if k_gram exists in string:
                sig(s, i) = min(sig(s,i), h_i(row))
    """
    row_index = -1
    for random_k_mer in random_grams:
        row_index += 1
        # For each row representing a random chosen k-mer h_i(r) should be computed

        for key in k_gram_dictionary:  # for each read
            if random_k_mer in k_gram_dictionary[key]:  # we have one in this row for the read
                for hash_function_index in range(n):
                    a, b = hash_functions[hash_function_index]
                    h_i_row = (a * row_index + b) % r

                    signature_dictionary[hash_function_index][key] = min(signature_dictionary[hash_function_index][key],
                                                                         h_i_row)

    LSH_dict = {}
    for read_id in read_id_list:
        LSH_dict[read_id] = []

    b = 5
    m = 4
    for read_id in read_id_list:
        integer_list = []  # it should have m elements
        hash_function_counter = 0

        for hash_function_index in signature_dictionary:

            integer_list.append(signature_dictionary[hash_function_index][read_id])
            hash_function_counter += 1

            if hash_function_counter == m:
                LSH_dict[read_id].append(bandHashFunction(integer_list))
                hash_function_counter = 0
                integer_list = []

        print(LSH_dict[read_id])


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

    print(random_grams_counter)


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

    hash_function_pairs = set()
    while len(hash_function_pairs) < n:
        hash_function_pairs.add((random.randint(1, p - 1), random.randint(1, p - 1)))

    return list(hash_function_pairs)


dataset = FastaIO.read_fasta_file('../files/reads50.fasta')
getMinHashFunctions(dataset, 6, 29, 20)
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
