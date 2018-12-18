from os import listdir
import sys

number_of_cpus = 16
number_of_parameters = len(sys.argv)
if number_of_parameters > 1:
    number_of_cpus = int(sys.argv[1])

cluster_folders = listdir("Clusters")
number_of_clusters = len(cluster_folders)

S = ""
for i in range(number_of_clusters):
    current_cluster = cluster_folders[i]
    S += "mpirun -np " + str(number_of_cpus) + " ./conv " + str(current_cluster) + "\n"
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


filename = "run_convex.sh"
f = open(filename, "w+")
f.write(S)
f.close()
