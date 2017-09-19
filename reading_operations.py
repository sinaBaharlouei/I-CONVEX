from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser
from sklearn.cluster import KMeans
import numpy as np

def create_random_k_shingles():
    shingle_dict = {}


# reading a Fasta file
dataSet = SeqIO.parse('files/reads50.fasta', 'fasta')
np_dataset = np.array(dataSet)
print np_dataset

"""
for item in np_dataset:
    print (item)
    break
"""

# k_means = KMeans(n_clusters=100)
# k_means.fit(dataSet)


