#!/bin/bash
# Input is working directory of the genomic files
cd $1
mkdir -p after_liftover

# Check if liftOver tool exists, download if not
if [ ! -f "liftOver" ] || [ ! -x "liftOver" ]; then
    echo "Downloading liftOver tool..."
    wget http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver
    chmod +x liftOver
else
    echo "liftOver tool already exists, skipping download."
fi

# Check if chain file exists, download if not
if [ ! -f "hg18ToHg38.over.chain.gz" ]; then
    echo "Downloading hg18ToHg38 chain file..."
    wget http://hgdownload.soe.ucsc.edu/goldenPath/hg18/liftOver/hg18ToHg38.over.chain.gz
else
    echo "hg18ToHg38 chain file already exists, skipping download."
fi

# Perform liftOver operations
./liftOver ./before_liftover/adni_1.bed hg18ToHg38.over.chain.gz ./after_liftover/ADNI_1_Hg38.bed ./after_liftover/unlifted_ADNI_1_Hg38.bed
./liftOver ./before_liftover/adni_2_1.bed hg18ToHg38.over.chain.gz ./after_liftover/ADNI_2_1_Hg38.bed ./after_liftover/unlifted_ADNI_2_1_Hg38.bed
./liftOver ./before_liftover/adni_2_2.bed hg18ToHg38.over.chain.gz ./after_liftover/ADNI_2_2_Hg38.bed ./after_liftover/unlifted_ADNI_2_2_Hg38.bed
./liftOver ./before_liftover/adni_3_1.bed hg18ToHg38.over.chain.gz ./after_liftover/ADNI_3_1_Hg38.bed ./after_liftover/unlifted_ADNI_3_1_Hg38.bed
./liftOver ./before_liftover/adni_3_2.bed hg18ToHg38.over.chain.gz ./after_liftover/ADNI_3_2_Hg38.bed ./after_liftover/unlifted_ADNI_3_2_Hg38.bed