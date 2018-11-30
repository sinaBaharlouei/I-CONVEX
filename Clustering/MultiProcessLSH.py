#!/usr/bin/python
#SBATCH -n 20      # tasks requested
#SBATCH --mem=64G  # Memory
#SBATCH -t 24:00:00  # time requested in hour:minute:second

import csv
import timeit
import multiprocessing as mp


def generateGraph(current_bands, ind):
    pairs_in_graph = set()
    print("Hi", ind)
    t_0 = timeit.default_timer()

    number_of_permutations = 10

    for current_band in current_bands:
        L = len(current_band)
        if L < 2:
            continue
        t = 1
        while L > t and t < number_of_permutations:
            for j in range(L - t):
                if current_band[j] < current_band[j + t]:
                    pairs_in_graph.add((current_band[j], current_band[j + t]))
                else:
                    pairs_in_graph.add((current_band[j + t], current_band[j]))
            t += 1

    t_1 = timeit.default_timer()

    print("Creating time for ", ind, ": ", t_1 - t_0)

    with open("batch" + str(ind) + '.csv', 'wb') as f:
        w = csv.writer(f, delimiter=',')
        for edge in pairs_in_graph:
            w.writerow(edge)

    t_2 = timeit.default_timer()
    print("Writing time for ", ind, ": ", t_2 - t_1)


# Collect the matrix
LSH_dict = {}
bands_dict = {}

number_of_processors = 20

with open('hash_functions.csv', 'r') as csvfile:
    pairs = list(csv.reader(csvfile, delimiter=','))
    r = int(pairs[0][0])
    b = int(pairs[0][1])
    k_parameter = int(pairs[0][2])
    number_of_chunks = int(pairs[0][3])
    print(number_of_chunks)

t0 = timeit.default_timer()
for i in range(number_of_chunks):
    with open('LSH_' + str(i + 1) + '.csv') as csvfile:
        rows = list(csv.reader(csvfile, delimiter=','))
        print(rows[2][0])
        for row in rows:
            row_len = len(row)
            LSH_dict[row[0]] = []
            for k in range(b):
                bands_dict[(k + 1, row[k + 1])] = []  # keys: (column, value)
                LSH_dict[row[0]].append(row[k + 1])

t1 = timeit.default_timer()
print(t1 - t0)

# Create graph
for key in LSH_dict:
    band_values = LSH_dict[key]
    for i in range(len(band_values)):
        bands_dict[(i + 1, band_values[i])].append(key)

bands_list = list(bands_dict.values())

bands_len = len(bands_dict)
batch_size = bands_len // number_of_processors

if __name__ == "__main__":
    # Define an output queue
    my_q = mp.Queue()
    jobs = []

    for i in range(number_of_processors - 1):
        start = i * batch_size
        end = start + batch_size
        current_batch = bands_list[start:end]

        p = mp.Process(target=generateGraph, args=(current_batch, i + 1,))
        jobs.append(p)

    start = (number_of_processors - 1) * batch_size
    end = bands_len
    current_batch = bands_list[start:end]
    p = mp.Process(target=generateGraph, args=(current_batch, number_of_processors,))
    jobs.append(p)

    for p in jobs:
        p.start()

    # Exit the completed processes
    for p in jobs:
        p.join()
