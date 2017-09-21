import random

import binascii


def find_k_grams(record, k):
    k_gram_array = []
    length = len(record)
    for i in range(length - k):
        # k_gram_array.append(binascii.crc32(bytes(str(record[i:i + k]), 'ascii')) & 0xffffffff)
        k_gram_array.append(record[i:i + k])
    return k_gram_array


def generate_all_k_grams(k, letter_set):
    if k == 1:
        return letter_set

    final_set = []
    previous_set = generate_all_k_grams(k - 1, letter_set)
    for item in previous_set:
        for letter in letter_set:
            final_set.append(item + letter)
    return final_set


def get_random_k_grams(token_set, number_of_items):
    final_set = []
    indices = random.sample(range(len(token_set)), number_of_items)
    for index in indices:
        # final_set.append(binascii.crc32(bytes(token_set[index], 'ascii')) & 0xffffffff)
        final_set.append(token_set[index])

    print(final_set)
    return final_set

