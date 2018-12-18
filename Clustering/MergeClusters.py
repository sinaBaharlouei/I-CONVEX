import csv

keys = {}
with open('FinalClusters.csv', 'r') as csvfile:
    cluster_ids = list(csv.reader(csvfile, delimiter=','))
    for item in cluster_ids:
        keys[int(item[1])] = 0

    for item in cluster_ids:
        keys[int(item[1])] += 1

print(len(keys))
minimum = min(keys.keys())
min_pos = abs(minimum)
print(min_pos)
print(keys)
size = sorted(keys.values(), reverse=True)
print(size)

threshold = 5

for i in range(len(cluster_ids)):
    if keys[int(cluster_ids[i][1])] < threshold:
        cluster_ids[i][1] = 0
    else:
        cluster_ids[i][1] = int(cluster_ids[i][1]) + min_pos + 1


with open('MergedClusters.csv', 'wb') as f:  # Just use 'w' mode in 3.x
    print("Write to file ... ")
    w = csv. writer(f, delimiter=',')
    for item in cluster_ids:
        w.writerow([item[0], item[1]])

keys = {}
for item in cluster_ids:
    keys[int(item[1])] = 0

for item in cluster_ids:
    keys[int(item[1])] += 1

print(len(keys))
minimum = min(keys.keys())
min_pos = abs(minimum)
print(minimum)
print(keys)
size = sorted(keys.values(), reverse=True)
print(size)

