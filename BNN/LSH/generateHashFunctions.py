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


print("This is the name of the script: ", sys.argv[0])
number_of_parameters = len(sys.argv)

if number_of_parameters > 2:
    r = int(sys.argv[1])
    b = int(sys.argv[2])

else:
    r = 1
    b = 10

n = r * b

hash_functions = createHashFunctions(n, 1000)
with open('hash_functions.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.writer(f, delimiter=',')
    w.writerow([r, b])
    for item in hash_functions:
        w.writerow([item[0], item[1]])
