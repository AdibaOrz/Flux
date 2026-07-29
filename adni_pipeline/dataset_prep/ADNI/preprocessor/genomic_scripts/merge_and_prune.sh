#!/bin/bash
working_dir=$1

# Copy merge script to current directory
cp ./genomic_scripts/merge_genomic.sh ${working_dir}
cd ${working_dir}
export PATH=/usr/local/bin:$PATH

# Run merge script
chmod +x ./merge_genomic.sh
./merge_genomic.sh

# Run PLINK for pruning
plink --bfile ./liftovered/ADNI_merged --indep-pairwise 50 5 0.1 --out ./liftovered/ADNI_merged_pruned
plink --bfile ./liftovered/ADNI_merged --extract ./liftovered/ADNI_merged_pruned.prune.in --make-bed --out ./liftovered/ADNI_final