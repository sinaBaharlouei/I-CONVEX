import os
from Bio import SeqRecord
from Bio import SeqIO
from Bio import Seq

array = []
cluster_keys = os.listdir("Clusters")
print("Number of clusters: ", len(cluster_keys))

for item in cluster_keys:
    with open('Clusters/' + item + '/ColapsedCentroid.txt', 'r') as trFile:
        content = trFile.read().splitlines()
        array += content

final_transcripts = open('final_transcripts.txt', 'w')
for item in array:
    if len(item) > 0:
        final_transcripts.write("%s\n" % item)


i = 0
records = []
for item in array:

    new_id = 's_' + str(i)
    new_seq = SeqRecord.SeqRecord(seq=Seq.Seq(item), id=new_id)
    records.append(new_seq)
    i += 1

print("Total number of recovered transcripts: ", i)
SeqIO.write(records, "final_transcripts.fasta", "fasta")
