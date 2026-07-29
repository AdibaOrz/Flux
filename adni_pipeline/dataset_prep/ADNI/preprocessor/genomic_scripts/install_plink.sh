#!/bin/bash
# This script installs PLINK and sets it up for use in the working directory

working_dir=$1
cd ${working_dir}

# Check if PLINK.zip is already downloaded
if [ ! -f "plink_linux_x86_64_20241022.zip" ]; then
    echo "Downloading PLINK..."
    wget https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20241022.zip
else
    echo "PLINK zip file already exists, skipping download."
fi
unzip plink_linux_x86_64_20241022.zip -d plink_install
cd plink_install
sudo cp plink /usr/local/bin
chmod 755 /usr/local/bin/plink