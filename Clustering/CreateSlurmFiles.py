from os import listdir
import sys

batch_size = 20
number_of_parameters = len(sys.argv)
if number_of_parameters > 1:
    batch_size = int(sys.argv[1])

cluster_folders = listdir("Clusters")
number_of_clusters = len(cluster_folders)

number_of_batches = number_of_clusters // batch_size + 1

for i in range(number_of_batches - 1):
    start = i * batch_size
    end = (i + 1) * batch_size
    S = "#!/bin/bash\n"
    S += "#SBATCH --ntasks=16\n"
    S += "#SBATCH --time=24:00:00\n"
    S += "#SBATCH --export=NONE\n"
    S += "#SBATCH --mem=64G\n"

    S += "timestamp1=$(date +%s)\n"
    S += "source /usr/usc/openmpi/default/setup.sh\n"
    for j in range(start, end):
        current_cluster = cluster_folders[j]
        S += "mpirun -np 16 ./conv " + str(current_cluster) + "\n"
        S += "./post " + str(current_cluster) + "\n"
        S += "cd Clusters/" + str(current_cluster) + "\n"
        S += "rm -rf ActLength.txt\n"
        S += "rm -rf BreakReads\n"
        S += "rm -rf CentroidIdRevised.txt\n"
        S += "rm -rf CentroidId.txt\n"
        S += "rm -rf Centroids.Fasta\n"
        S += "rm -rf ClusterEndAlign.txt\n"
        S += "rm -rf ClusterNormalScore.txt\n"
        S += "rm -rf ClusterReadsId.txt\n"
        S += "rm -rf dist\n"
        S += "rm -rf dimensions.txt\n"
        S += "rm -rf EndAlign.txt\n"
        S += "rm -rf Filtering\n"
        S += "rm -rf FinalCenters.txt\n"
        S += "rm -rf FinalCentroids.txt\n"
        S += "rm -rf FinalRho.txt\n"
        S += "rm -rf FinishCode.txt\n"
        S += "rm -rf IMS\n"
        S += "rm -rf matchedCenters.txt\n"
        S += "rm -rf NormalScores.txt\n"
        S += "rm -rf Output\n"
        S += "rm -rf OutputBest\n"
        S += "rm -rf reads\n"
        S += "rm -rf ReadCentroidAssociation.txt\n"
        S += "rm -rf ReadsCentroidID.txt\n"
        S += "rm -rf ReadsNoisy.txt\n"
        S += "rm -rf readsOrig\n"
        S += "rm -rf Remaining\n"
        S += "rm -rf Scores.txt\n"
        S += "cd ../..\n"
    S += "timestamp2=$(date +%s)\n"
    S += "total_time=`expr $timestamp2 - $timestamp1`\n"
    S += "echo $total_time\n"

    filename = "cluster_batch" + str(i+1) + '.sh'
    f = open(filename, "w+")
    f.write(S)
    f.close()

start = (number_of_batches - 1) * batch_size
end = number_of_clusters
S = "#!/bin/bash\n"
S += "#SBATCH --ntasks=16\n"
S += "#SBATCH --time=24:00:00\n"
S += "#SBATCH --export=NONE\n"
S += "#SBATCH --mem=64G\n"

S += "timestamp1=$(date +%s)\n"
S += "source /usr/usc/openmpi/default/setup.sh\n"
for j in range(start, end):
    current_cluster = cluster_folders[j]
    S += "mpirun -np 16 ./conv " + current_cluster + "\n"
    S += "./post " + current_cluster + "\n"
    S += "cd Clusters/" + str(current_cluster) + "\n"
    S += "rm -rf ActLength.txt\n"
    S += "rm -rf BreakReads\n"
    S += "rm -rf CentroidIdRevised.txt\n"
    S += "rm -rf CentroidId.txt\n"
    S += "rm -rf Centroids.Fasta\n"
    S += "rm -rf ClusterEndAlign.txt\n"
    S += "rm -rf ClusterNormalScore.txt\n"
    S += "rm -rf ClusterReadsId.txt\n"
    S += "rm -rf dist\n"
    S += "rm -rf dimensions.txt\n"
    S += "rm -rf EndAlign.txt\n"
    S += "rm -rf Filtering\n"
    S += "rm -rf FinalCenters.txt\n"
    S += "rm -rf FinalCentroids.txt\n"
    S += "rm -rf FinalRho.txt\n"
    S += "rm -rf FinishCode.txt\n"
    S += "rm -rf IMS\n"
    S += "rm -rf matchedCenters.txt\n"
    S += "rm -rf NormalScores.txt\n"
    S += "rm -rf Output\n"
    S += "rm -rf OutputBest\n"
    S += "rm -rf reads\n"
    S += "rm -rf ReadCentroidAssociation.txt\n"
    S += "rm -rf ReadsCentroidID.txt\n"
    S += "rm -rf ReadsNoisy.txt\n"
    S += "rm -rf readsOrig\n"
    S += "rm -rf Remaining\n"
    S += "rm -rf Scores.txt\n"
    S += "cd ../..\n"
S += "timestamp2=$(date +%s)\n"
S += "total_time=`expr $timestamp2 - $timestamp1`\n"
S += "echo $total_time\n"
filename = "cluster_batch" + str(number_of_batches) + '.sh'
f = open(filename, "w+")
f.write(S)
f.close()


S = ""
for j in range(number_of_batches):
    S += "sbatch cluster_batch" + str(j+1) + ".sh\n"

filename = "run_convex.sh"
f = open(filename, "w+")
f.write(S)
f.close()