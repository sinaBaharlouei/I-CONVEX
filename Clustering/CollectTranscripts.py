import os

array = []
cluster_keys = os.listdir("Clusters")
print(len(cluster_keys))

for item in cluster_keys:
    with open('Clusters/' + item + '/ColapsedCentroid.txt', 'r') as trFile:
        content = trFile.read().splitlines()
        array += content

print("total transcripts: ", len(array))
final_transcripts = open('final_transcripts.txt', 'w')
for item in array:
    if len(item) > 0:
        final_transcripts.write("%s\n" % item)

