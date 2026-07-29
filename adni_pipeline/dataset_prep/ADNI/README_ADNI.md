# ADNI
Alzheimer's Disease Neuroimaging Initiative (ADNI) is a longitudinal multi-center, observational study dataset.
The study data includes various modalities such as imaging (e.g., MRI, PET), clinical assessments, genetics & bio fluid and more.
For full description and access details, please refer to the official [ADNI website](http://adni.loni.usc.edu/).

In our experiments, we focus on the following modalities: `imaging (MRI)`, `genomic`, `biospecimen`, and `clinical`.


## 0. Data Access & Necessary files
- ADNI requires credentialed access through the ADNI website
- Go to the [ADNI website](http://adni.loni.usc.edu/) and apply for data access
- After obtaining access, you will be able to download the data from the ADNI website
- First, create the following directory structure to download the data:
  ```
     datasets/
         ├── ADNI/
               ├── raw/
                   ├── diagnosis/               # <- download dir for diagnosis data (labels)
                   ├── MRI/                     # <- download dir for raw MRI data
                   ├── genomic/                 # <- download dir for raw genomic data
                   ├── biospecimen/             # <- download dir for raw biospecimen data
                   ├── clinical/                # <- download dir for raw clinical data
               ├── processed/                   # <- processed data will be saved here
               ├── partitioned/                 # <- partitioned data will be saved here
   ```

### Configure local paths

Copy [`adni_paths_example.json`](adni_paths_example.json) to `adni_paths.json`
in this directory and update the paths to your local layout—the same pattern as
[`data_preprocessor/dataset_paths_example.json`](../../../data_preprocessor/dataset_paths_example.json)
→ `dataset_paths.json` for the wearable datasets. Preprocessing, partitioning,
validation, and training read this file by default. It defines:

- `adni_root`: root containing `raw/`, `processed/`, and `partitioned/`
- `downloaded_date`: suffix used by downloaded ADNI CSV files
- `mri_reference_atlas`: path to the MNI152 reference atlas
- `mri_rename_source_dir`: downloaded MRI directory used by `rename_MRI_dirnames.py`

## 1. Diagnosis File
- This is files serves as the ground truth label for this dataset
- After obtaining access for ADNI and logging in, go to Search & Download > ARC Builder, then add these files into collections and download
    1. DXSUM
- Place it into your diagnosis data directory, this file will be used by the preprocessing steps below


## 2. MRI
### 2.1 MRI: Overview
- MRI image processing is done based on the [following Github repository](https://github.com/SANCHES-Pedro/adni_preprocessing).
- Preprocessing steps involve:
  1. Reorientation (using FSL `fslreorient2std`)
  2. Cropping (using FSL `robustfov`)
  3. Brain extraction (using FSL `bet`)
  4. Atlas affine registration (using FSL `flirt`)
     - We use resolution 2mm. Resolution can be specified in `preprocessing_configs.py`
  5. Structure segmentation and bias correction (using FSL `fast`)
  6. Intensity normalization between [-1, 1]
      - Clip intensity at 99.5% of max values
  7. Central slice cropping
  8. Save processed image as torch tensor (other options: 2D images, numpy array)


### 2.2 MRI: How to run?
1. [**Install FSL**] Since most of the MRI preprocessing steps are done using **FSL tool** (library of analysis tools for MRI, FMRI, and diffusion brain imaging data), make sure to install it first.
   - For installation instructions, please refer to the [FSL website](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index).
   - After installation is complete, restart your terminal.
   - Activate fsl environment, where `$FSLDIR` is the path to your FSL installation.
     ```bash
     source $FSLDIR/etc/fslconf/fsl.sh
     ```

2. [**Install dependent libraries**] Install a conda environment if you haven't already.
   ```bash
   conda env create -f medmfl.yml
   ``` 

3. [**Set up paths and configs**] Configure local paths as described above.
   Non-path MRI options (output format, axial size, reprocessing, and W&B
   logging) remain in `preprocessor/preprocessing_configs.py`.

4. [**Run preprocessing**] Run the preprocessing script to process the MRI data.
   ```bash
   python preprocess_MRIs.py
   ```
5. [**Remove unusable processed files**] Run the following script to remove files with NaN values, generated as a result of preprocessing.
    ```bash
   python validate_MRIs.py
   ```



## 3. Genomic
### 3.1 Genomic: Overview
- Genomic data processing is a modified pipeline based on [Flex-MoE](https://github.com/UNITES-Lab/Flex-MoE/tree/main/data/adni) processing pipeline, and we will be using ADNI1, ADNI2, and ADNI3 genotype data sets.
- The processing pipeline make use of PLINK, which an open-source genome association analysis toolkit. The steps taken includes removing SNPs with high correlation with each other, merging them to find the multi-variant SNPs (chromosomes that have 3+ alleles), then doing a second pass to exclude the multi-variant SNPs and finally merging them together

### 3.2 Genomic: How to run?
1. [**Downloading the data**] After obtaining access for ADNI and logging in, go to Search & Dowwnload > Genetic Files, then search and download these files
    1. ADNI 1 SNP genotype data - PLINK
    2. ADNI GO/2 SNP genotype data - Complete PLINK for sets 1 - 9
    3. ADNI GO/2 SNP genotype data - Complete PLINK for sets 10 - 15
    4. ADNI3 SNP genotype data Set 1 - PLINK
    5. ADNI3 SNP genotype data Set 2 - PLINK


2. [**Preparing the Files**] Unzip/Untar the files, there should be 3 files (*.bed, *.bim, *.fam) after decompressing


3. [**Set up paths and configs**] Configure local paths as described above.
   Genomic inputs are expected under `ADNI/raw/genomic/` with the filenames
   used in `preprocessor/preprocessing_configs.py`.

4. [**Run preprocessings**] Run the preprocessing script to process the MRI data.
   ```bash
   python preprocess_genomic.py
   ```

 

## 4. Biospecimen
### 4.1 Biospecimen: Overview
- Biospecimen data processing is a modified pipeline based on [Flex-MoE](https://github.com/UNITES-Lab/Flex-MoE/tree/main/data/adni) processing pipeline. These data are structured as tabular data.

### 4.2 Biospecimen: How to run?
1. [**Downloading the data**] After obtaining access for ADNI and logging in, go to Search & Dowwnload > ARC Builder, then add these files into collections and download
    1. APOERES
    2. UPENNBIOMK_ROCHE_ELECSYS


2. [**Preparing the Files**] Unzip the files and move them into your biospecimen data directory


3. [**Set up paths and configs**] Configure local paths as described above.
   Biospecimen CSVs are expected under `ADNI/raw/biospecimen/`.

4. [**Run preprocessings**] Run the preprocessing script to process the MRI data.
   ```bash
   python preprocess_biospecimen.py
   ```


## 5. Clinical
### 5.1 Clinical: Overview
- Biospecimen data processing is a modified pipeline based on [Flex-MoE](https://github.com/UNITES-Lab/Flex-MoE/tree/main/data/adni) processing pipeline. These data are structured as tabular data.

### 5.2 Clinical: How to run?
1. [**Downloading the data**] After obtaining access for ADNI and logging in, go to Search & Dowwnload > ARC Builder, then add these files into collections and download
    1. MEDHIST
    2. NEUROEXM
    3. PTDEMOG
    4. RECCMEDS
    5. VITALS


2. [**Preparing the Files**] Unzip the files and move them into your clinical data directory


3. [**Set up paths and configs**] Configure local paths as described above.
   Clinical CSVs are expected under `ADNI/raw/clinical/`.

4. [**Run preprocessings**] Run the preprocessing script to process the MRI data.
   ```bash
   python preprocess_clinical.py
   ```

## 6. Partitioning and missingness simulation

After all modalities are preprocessed, create federated train/validation/test
partitions from the `partitioner/` directory. The public implementation supports
the `stratified` and `modality_unique` splits used in the experiments. Simulated
missingness partitions are derived from the stratified split.

```bash
cd partitioner
python create_partitions.py --split stratified --num_silos 5
```

To create a simulated-missingness variant of the default stratified partition:

```bash
python simulate_missingness.py \
    --availability avail_med \
    --p_intra 0.3 \
    --seed 42
```

Then impute, scale, and save the tabular arrays used by ADNI training:

```bash
python process_partitioned_data.py \
    --modality clinical \
    --mode federated
```

`run_process_partitioned_data.sh` runs this final step over its configured
partition list using the same local path configuration.
