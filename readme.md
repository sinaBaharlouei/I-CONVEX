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

### Map Candidate Similar Pairs to the Same Bucket
Using MinHash signatures as a base of a family of Locality Sensitive Hash, we can map the similar candidate pairs of noisy reads into to the same bucket. Thus, after running 
**MultiProcessLSH.py** the candidate similar pairs will be available in the **batch1.csv** to **batchn.csv** files. To obtain these files you should run the following command:
```
ClusteringReads/Clustering$ sbatch MultiProcessLSH.py
```
If you do not have access to an HPC server, you can run it as follows:
```
ClusteringReads/Clustering$ python MultiProcessLSH.py
```

### Collect All of the Candidate Pairs
In this step, you should run ./collector.sh to obtain the final candidate pairs. This script will merge all **batch*.csv** files and will remove the duplicate rows.
The output will be G.csv file.
```
ClusteringReads/Clustering$ ./collector.sh
```

1) Fasta file trimming: python FastaPreparation.py
6) Validate Candidate Pairs: python ValidateCandidatePairs.py
7) Cluster the polished graph: python Clustering.py
8) Merge small clusters: python MergeClusters.py
9) Create cluster directories as the input of CONVEX: python CreateClusterDirectories.py