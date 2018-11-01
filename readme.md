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
ClusteringReads/Clustering$ python SplitFile.py
```
After running, there should be a **hash_functions.csv** file and **chunk1.fasta** to **chunkn.fasta** files as the outputs.

### Find MinHash Signatures:
To find MinHash signatures, you should run **MPMH.py** file. Since **MPMH.py** uses multi processing to enhance the speed, 
it is recommended to run it on a High Performance Computing Server(HPC) as follows:
```
ClusteringReads/Clustering$ sbatch MPMH.py
```
Otherwise, you can run it directly with python. Note that if the number of chunks is more than 20, you need to run MPMH.py for each batch of 20 chunks separately.
For example if SplitFile.py generates 50 files chunk1.fasta to chunk50.fasta (2.5 Million reads) you need to run MPHM three times as follows:
```
ClusteringReads/Clustering$ sbatch MPMH.py
ClusteringReads/Clustering$ sbatch MPMH.py 2
ClusteringReads/Clustering$ sbatch MPMH.py 3
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