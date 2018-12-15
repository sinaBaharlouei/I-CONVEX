cat batch*.csv > merged.csv
python RemoveDuplicatePairs.py
rm -rf batch*.csv
rm -rf LSH*.csv
rm -rf chunk*.fasta
