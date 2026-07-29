"""
References go to: https://github.com/UNITES-Lab/Flex-MoE
Some parts have been modified to fit the current project
"""

import os
import anndata
import subprocess
import numpy as np
import pandas as pd
from preprocessing_configs import get_config_dict
from pandas_plink import read_plink1_bin, write_plink1_bin



def main():
    # Get configuration dictionary
    config = get_config_dict(modality='genomic')
    
    # Set paths
    adni_1_path = str(config['adni_1_path'])
    adni_2_1_path = str(config['adni_2_1_path'])
    adni_2_2_path = str(config['adni_2_2_path'])
    adni_3_1_path = str(config['adni_3_1_path'])
    adni_3_2_path = str(config['adni_3_2_path'])
    working_dir = str(os.path.dirname(config["adni_1_path"]))

    # Prepare liftover files
    adni_paths = [adni_1_path, adni_2_1_path, adni_2_2_path, adni_3_1_path, adni_3_2_path] 

    for i in range(5):
        df = pd.read_csv(adni_paths[i] + '.bim', header=None, sep='\t')
    
        valid_bases = ['A', 'C', 'G', 'T']
        df_filtered = df[df[4].isin(valid_bases) & df[5].isin(valid_bases)]
        df_filtered = df_filtered.reset_index(drop=True)
        df_filtered.to_csv(adni_paths[i] + '.bim', index=False, header=None, sep='\t')

    save_paths = ['adni_1.bed', 'adni_2_1.bed', 'adni_2_2.bed', 'adni_3_1.bed', 'adni_3_2.bed']
    os.makedirs(working_dir + '/before_liftover/', exist_ok=True)
    os.makedirs(working_dir + '/after_liftover/', exist_ok=True)
    os.makedirs(working_dir + '/liftovered', exist_ok=True)

    for i in range(len(adni_paths)):
        bim_path = adni_paths[i]
        save_path = working_dir + '/before_liftover/' + save_paths[i]
        tmp = pd.read_csv(f'{bim_path}.bim', header=None, sep='\t')
        
        chr_id = tmp[0].replace({23: 'X', 24: 'Y'}).where((0 < tmp[0]) & (tmp[0] < 25), None)
        
        filtered_data = tmp[chr_id.notna()]
        
        new_tmp = pd.DataFrame({
            '0': 'chr' + chr_id[chr_id.notna()].astype(str),
            '1': filtered_data[3],
            '2': filtered_data[3],
            '3': filtered_data[1]
        })
        
        new_tmp = new_tmp.reset_index(drop=True)
        new_tmp.to_csv(save_path, index=False, sep='\t', header=None)

    # Liftover
    os.system("chmod +x ./genomic_scripts/liftover.sh")
    os.system(f"./genomic_scripts/liftover.sh {working_dir}")

    # Filtering
    after_liftover_paths = ['/after_liftover/ADNI_1', '/after_liftover/ADNI_2_1', '/after_liftover/ADNI_2_2', '/after_liftover/ADNI_3_1', '/after_liftover/ADNI_3_2']

    for after_liftover_path in after_liftover_paths:
        _tmp = pd.read_csv(working_dir + after_liftover_path + '_Hg38.bed', header=None, sep='\t')
        _tmp_filtered = _tmp[~_tmp[0].str.contains('_')]
        _tmp_filtered.to_csv(working_dir + after_liftover_path + '_Hg38.bed', index=False, header=False, sep='\t')

    # Align Bin Bed Fam with Liftover
    adni_paths = [adni_1_path, adni_2_1_path, adni_2_2_path, adni_3_1_path, adni_3_2_path]
    liftover_paths = ['/after_liftover/ADNI_1', '/after_liftover/ADNI_2_1', '/after_liftover/ADNI_2_2', '/after_liftover/ADNI_3_1', '/after_liftover/ADNI_3_2']
    output_paths = ['/liftovered/ADNI_1', '/liftovered/ADNI_2_1', '/liftovered/ADNI_2_2', '/liftovered/ADNI_3_1', '/liftovered/ADNI_3_2']

    for i in range(5):
        adni_path = adni_paths[i]
        liftover_path = working_dir + liftover_paths[i] + '_Hg38.bed'
        output_path = working_dir + output_paths[i]
        
        G = read_plink1_bin(adni_path+'.bed', adni_path+'.bim', adni_path+'.fam')

        # Read the liftover file to get the new SNP IDs
        liftover = pd.read_csv(liftover_path, sep='\t', header=None, names=['chr', 'pos1', 'pos2', 'id'])

        # Find indices of the SNPs in bim that are present in the liftover file
        indices = np.arange(G.shape[1])[(pd.DataFrame(G.snp)[0].isin(liftover['id']))]

        if i == 0:
            fam = pd.read_csv(
                adni_1_path + '.fam',
                sep=r'\s+',
                header=None,
                names=['fid', 'iid', 'father', 'mother', 'gender', 'trait'],
            )
            fam_indices = ~fam['iid'].isin(['073_S_0909', '130_S_1201'])
            _G = G[fam_indices, indices]

        else:
            _G = G[:, indices]
        
        _G['chrom'] = ('variant', np.array(liftover['chr']))
        _G['pos'] = ('variant', np.array(liftover['pos1']))
        write_plink1_bin(_G, output_path+'.bed', output_path+'.bim', output_path+'.fam')

    G_list = []
    for i in range(5):
        adni_path = working_dir + output_paths[i]
        
        G = read_plink1_bin(adni_path+'.bed', adni_path+'.bim', adni_path+'.fam')
        G_list.append(G)

    # Install Plink
    os.system("chmod +x ./genomic_scripts/install_plink.sh")
    os.system(f"sudo ./genomic_scripts/install_plink.sh {working_dir}")
    
    # Merge and Prune using Plink
    os.system("chmod +x ./genomic_scripts/merge_and_prune.sh")
    os.system(f"sudo ./genomic_scripts/merge_and_prune.sh {working_dir}")

    # Save
    # save_dir = str(os.path.dirname(config['save_dir']))
    save_dir = str(config['save_dir'])
    os.makedirs(save_dir, exist_ok=True)

    save_path = save_dir + "/genomic_merged.h5ad"
    merged_path = working_dir + "/liftovered/ADNI_final"
    G_merge = read_plink1_bin(merged_path+'.bed', merged_path+'.bim', merged_path+'.fam')

    genotype_df = G_merge.to_pandas()
    genotype_df.columns = G_merge.snp

    adata = anndata.AnnData(X=genotype_df.values,
                            obs=pd.DataFrame(index=genotype_df.index),
                            var=pd.DataFrame(index=genotype_df.columns))

    adata.write_h5ad(save_path)


if __name__ == "__main__":
    main()
