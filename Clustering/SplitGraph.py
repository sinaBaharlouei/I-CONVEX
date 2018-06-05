import csv


with open("MGK10R1B10P10.csv", 'r') as csvfile:
    pairs = list(csv.reader(csvfile, delimiter=','))

    number_of_pairs = len(pairs)
    number_of_batches = number_of_pairs // 10000000

    print(number_of_batches)

    for i in range(number_of_batches):
        start = i * 10000000
        end = i * 10000000 + 10000000
        current_pairs = pairs[start:end]
        with open("G" + str(i+1) + ".csv", 'wb') as resultFile:
            wr = csv.writer(resultFile)
            wr.writerows(current_pairs)

    # Last File
    start = number_of_batches * 10000000
    end = number_of_pairs
    if end > start:
        current_pairs = pairs[start:end]
        with open("G" + str(number_of_batches + 1) + ".csv", 'wb') as resultFile:
            wr = csv.writer(resultFile)
            wr.writerows(current_pairs)
