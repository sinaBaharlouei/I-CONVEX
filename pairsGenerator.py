from DatasetGenerator.dissimilarPairsGenerator import generateFile
from DatasetGenerator.similarPairsGenerator import generateSimilarPairs

# generateFile("dissimilar5k.csv", 100000, 400, 20, 20)
generateSimilarPairs("similar500K.csv", 400, 500000)
