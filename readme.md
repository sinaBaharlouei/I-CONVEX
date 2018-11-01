### Length of transcripts
To compare each pair of transcripts, EDIT distance is an appropriate measurement which is equal to the minimum number of insertion and deletions
converting one transcript to the another one. By using dynamic programming, EDIT distance could be computed; however, time complexity is quadratic in terms of
strings length. One remedy for this problem is to apply hash functions on transcript to compress them to shorter signatures.
We have used Binarized neural network and minHash signature for training appropriate hash functions.

### Large scale dataset
One basic assumption of the project is that the dataset is large scale and classic cluster algorithms are too slow and are not applicable.
We have used Locality sensitive hashing for reducing candidate pairs for comparison.

## Generating dataset
generateDataset.py
pairsGenerator.py

## Cluster Noisy Reads
Clustering of noisy reads before running CONVEX algorithm has several advantages over running CONVEX directly on the whole dataset. 
First, it decreases the order complexity of algorithm and eliminates its dependency to M, the number of transcripts (Centroids).
Moreover, it enables parallel running of CONVEX on different clusters. Clustering of a datsaet of noisy reads consists of the following steps:

### Split the Original File:
In the first step, the original file is splitted into the chunks, each of which contains 50K reads.
To run **SplitFile.py** make sure you change the name of original file to the **reads.fasta** and put it to the **Clustering** folder. Then run as follows:
```
baharlou@hpc-login3:/home/sina/ClusteringReads/Clustering$ python SplitFile.py
```

1) Fasta file trimming: python FastaPreparation.py
2) Split Fasta file into the chunks: python SplitFile.py
3) Obtain MinHash signatures: python MPMH.py
4) Find similar candidate pairs: python MultiProcessLSH.py
5) Collect all the candidate pairs from different processes: ./collector.sh
6) Validate Candidate Pairs: python ValidateCandidatePairs.py
7) Cluster the polished graph: python Clustering.py
8) Merge small clusters: python MergeClusters.py
9) Create cluster directories as the input of CONVEX: python CreateClusterDirectories.py