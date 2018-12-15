**CONVEX** is an iterative algorithm for solving "de Novo Transcriptome Recovery from long reads" problem. This algorithm starts with a set of short prefixes(4 or 5), and estimates the abundances of 
these prefixes based on the given noisy reads dataset. The abundance of these prefixes can be efficiently estimated by aligning them with the reads and solving a maximum likelihood estimation problem
through the expectation maximization (EM) algorithm. 
Therefore, all the high-abundant prefixes will be extended with one base(by adding either A, C, G, or T to the end of the each one of the prefixes with size L and obtaining four new prefixes with
size L+1) and the non-frequent ones will be truncated. This procedure continues until the complete recovery of all the transcripts. 

# Prerequisites
It is highly recommended to install the following packages via Anaconda. Download and install **Python 2.7 version** of Anaconda from [Anaconda, Python 2.7 Version](https://www.anaconda.com/download/#linux):
Then install the following packages via Conda:
* [Biopython](https://anaconda.org/anaconda/biopython)
* [Matplotlib](https://anaconda.org/conda-forge/matplotlib)
* [Igraph](https://anaconda.org/conda-forge/python-igraph)

## Install the GPU Version of Tensorflow
For validating the similarity of the detected candidate pairs, you need to install **Tensorflow GPU-version**. You can follow the instructions in [here](https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25)
to install it.

## Install the Message Passing Interface
To run CONVEX algorithm after the pre-clustering stage, you need to install Message Passing Interface(MPI) with the following command:
```
sudo apt install mpich
```


# Cluster Noisy Reads
Clustering of noisy reads before running CONVEX algorithm has several advantages over running CONVEX directly on the whole dataset.
First, it decreases the order complexity of the algorithm and eliminates its dependency to M, the number of transcripts (Centroids).
Moreover, it enables the parallel running of CONVEX on different clusters. 

## Run Pre-clustering (Basic Version)
If your input fasta file is not a large-scale one and you want to run it on your PC, follow the below instructions:
1. Move your input fasta file to the Clustering Folder and rename it to reads.fasta.
2. Run SplitFile.py to chunk the dataset:
    ```
    ClusteringReads/Clustering$ python SplitFile.py
    ```
3. Run the **commands.sh** file as follows:
    ```
    ClusteringReads/Clustering$ chmod 777 commands.sh
    ClusteringReads/Clustering$ ./commands.sh
    ```
At the end, the reads and their corresponding cluster identifiers will be written in **MergedClusters.csv**. Moreover, a folder **Clusters/** which contains subfolders each of which represents a cluster
will be created.

## Run Pre-clustering on a High-Performance Computing(HPC) Server (Advanced version)
Running Pre-clustering on an HPC server makes the pre-clustering part tremendously faster; However, it has more details compared to the basic version. 

### Split the Original File:
In the first step, the original file is split into the chunks, each of which contains 50K reads.
To run **SplitFile.py** make sure you change the name of the original file to the **reads.fasta** and put it to the **Clustering** folder. Then run as follows:
```
ClusteringReads/Clustering$ python SplitFile.py
```
After running, there should be a **hash_functions.csv** file and **chunk1.fasta** to **chunkn.fasta** files as the outputs.

### Find MinHash Signatures:
To find MinHash signatures, you should run **MPMH.py** file. Since **MPMH.py** uses multiprocessing to enhance the speed, 
it is recommended to run it on a High-Performance Computing Server(HPC) as follows:
```
ClusteringReads/Clustering$ sbatch MPMH.py
```
If an HPC server is not available, you can run it directly with python. Note that if the number of chunks is more than 20, you need to run MPMH.py for each batch of 20 chunks separately.
For example, if **SplitFile.py** generates 50 files **chunk1.fasta** to **chunk50.fasta** (2.5 Million reads) you need to run **MPHM.py** three times as follows:

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
The output will be the G.csv file.
```
ClusteringReads/Clustering$ ./collector.sh
```


### Validate Candidate Pairs with a Convolutional Neural Network
To validate the similarity of candidate pairs, a trained Convolutional Neural Network(CNN) should be executed on the obtained candidate pairs from the previous step. 

```
ClusteringReads/Clustering$ python FastaPreparation.py
ClusteringReads/Clustering$ python ValidatePairs.py
```
After running **ValidatePairs.py**, there should be a Net.csv file as the output.


### Clustering the Final Similarity Graph:
Now, we prepare to run the clustering algorithm on the final similarity graph:
```
ClusteringReads/Clustering$ python Clustering.py
```

And then, we should merge all the mini clusters with the size less than 5:
```
ClusteringReads/Clustering$ python MergeClusters.py
```

### Create Cluster Directories:
Finally, in order to run CONVEX on the different clusters, we should create a folder for each cluster:
```
ClusteringReads/Clustering$ python CreateClusterDirectories.py
```

# Running CONVEX on Pre-clusters:
After obtaining the pre-clusters, we are ready to run CONVEX on each cluster. First, we need to compile the following c files with MPI library:
```
ClusteringReads/Clustering$ mpicc -o conv CONVEXv16.c -lm
ClusteringReads/Clustering$ mpicc -o post PostProcessingV3.c
```


## Running CONVEX on HPC:
First, we need to run the following python script to create batches of clusters:

```
ClusteringReads/Clustering$ python CreateSlurmFiles.py 20
```
The parameter of this script is the number of clusters in each one of the batches. The default number is 20. 
Therefore, you should run the following script:
```
ClusteringReads/Clustering$ ./run_convex.sh
```
## Collecting the Final Transcripts:
To collect all the obtained transcripts from the different clusters, you need to run the following script:
```
ClusteringReads/Clustering$ python CollectTranscripts.py
```

The final transcripts will be saved in the **final_transcripts.txt** file.