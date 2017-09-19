import binascii
import timeit
from DataPreparation import FastaIO
from DataPreparation import KGrams
from DataPreparation.KGrams import find_k_grams
from Hashing import minHash

token_signature_array = []

i = 0
number_of_rows = 29  # Should be a prime number
number_of_hash_functions = 20
k = 6

# get random k-grams representing the rows of similarity matrix
# k-grams are hashed to 32bit integers for more convenience.
all_6_grams = KGrams.generate_all_k_grams(k, ['A', 'C', 'G', 'T'])
random_grams = KGrams.get_random_k_grams(all_6_grams, number_of_rows)

# Get hash functions
function_parameters = minHash.getMinHashFunctions(number_of_rows, number_of_hash_functions)
# print(function_parameters)

k_gram_dictionary = {}
n = 1

start = timeit.default_timer()

# get dataSet and create k-grams(key: record_id, value: set of k-grams)
dataset = FastaIO.read_fasta_file('files/reads50.fasta')
for record in dataset:
    k_gram_dictionary[record.id] = find_k_grams(record, k)
    if n > 10000:
        break
    n += 1

"""
# Create Signature dictionary: the key of each element would be the pair (S, i):
# S denotes the molecular string, i: denotes the i-th hash function.
# The value of the element with key(S, i) would be h_i(S)
signature_dictionary = {}

# First Sig(S, i) must be initialized to +inf
for record in dataset:
    for hash_function_index in range(number_of_hash_functions):
        signature_dictionary[(record.id, hash_function_index)] = 9999999

"""

# Algorithm Description:
"""
For each chosen k_gram(rows of matrix):
    1) compute h1(row), h2(row), ...
    2) for each string:
        if k_gram exist in string:
            sig(s, i) = min(sig(s,i), h_i(row))
"""
"""
for row in random_grams:
    # For each row representing a random chosen k-mer h_i(r) should be computed

    for key in k_gram_dictionary: # for each document

        if row in k_gram_dictionary[key]:

            hash_function_index = -1
            for hash_function_pair in function_parameters:

                a, b = hash_function_pair
                hash_function_index += 1
                h_i_row = (a * row + b) % number_of_rows

                signature_dictionary[(key, hash_function_index)] = min(signature_dictionary[(key, hash_function_index)], h_i_row)

stop = timeit.default_timer()

print(stop - start)
"""
"""
# For each molecular string in data set
for item in dataset:

    token_signature_array.append([])
    four_gram_array = KGrams.find_k_grams(item, 4)

    # for each k-gram of string:
    for token in four_gram_array:
        # Hash the token to a 32-bit integer.
        token_signature_array[i].append(binascii.crc32(token) & 0xffffffff)

    i += 1

    if i > 100:
        break
"""
