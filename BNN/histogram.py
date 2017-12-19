import matplotlib.pyplot as plt
import csv


my_list = []
with open('hash10.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    reads = list(reader)
    for item in reads:
        my_list.append(float(item[0]))
"""
print(reads[0])
print(reads[1])
print(my_list[0])
print(my_list[1])
"""
plt.hist(my_list, 20, facecolor='blue', alpha=0.5)
plt.show()