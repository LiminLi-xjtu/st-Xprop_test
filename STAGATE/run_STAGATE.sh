#!/bin/bash

# DLPFC
slices=('151507' '151508' '151509' '151510' '151669' '151670' '151671' '151672' '151673' '151674' '151675' '151676')
num_clusters=(7 7 7 7 5 5 5 5 7 7 7 7)

output_file="../evaluation/DLPFC/STAGATE.csv"
echo "slice,ari_mclust_na,ari_leiden_na,ari_louvain_na,ari_kmeans_na,ari_mclust,ari_leiden,ari_louvain,ari_kmeans" > "$output_file"

for i in "${!slices[@]}"; do
   slice=${slices[$i]}
   n_clusters=${num_clusters[$i]}
 echo "Running with slice=$slice"
 python run_DLPFC.py --slice "$slice" --n_clusters "$n_clusters" 2>/dev/null | tail -n 1 >> "$output_file"
done


#################################################################################################################


slices=('MBC' 'MBP')
for i in "${!slices[@]}"; do
    slice=${slices[$i]}
  echo "Running with slice=$slice"
  python run_MB.py --slice "$slice" 2>/dev/null | tail -n 1
done


##########################################################################################################

slices=('D4' 'D7' 'D10' 'D14')
# num_clusters=(20)


for i in "${!slices[@]}"; do
    slice=${slices[$i]}
    # n_clusters=${num_clusters[$i]}
  echo "Running with slice=$slice"
  python ./STAGATE/run_CHD.py --slice "$slice"
done

##########################################################################################################
slices=("A1" "B1" "C1" "D1" "E1" "F1" "G2" "H1")

output_file="../evaluation/HER2ST/STAGATE.csv"
echo "slice,ari_mclust,ari_leiden,ari_louvain,ari_kmeans" > "$output_file"

for i in "${!slices[@]}"; do
   slice=${slices[$i]}
 echo "Running with slice=$slice"
 python run_HER2ST.py --slice "$slice" 2>/dev/null | tail -n 1 >> "$output_file"
done