cat batch*.csv > merged.csv
uniq merged.csv > G.csv
rm -rf batch*.csv
rm -rf LSH*.csv
rm -rf chunk*.fasta
