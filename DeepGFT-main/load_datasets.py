import scanpy as sc
import numpy as np
import pandas as pd

def load_data(dataset_name, dataset_slice):

    
    if dataset_name=='DLPFC' or dataset_name=='MB' or dataset_name=='CHD' or dataset_name=='HER2ST' or dataset_name=='PDAC':
        load_path = f"../../datasets/{dataset_name}/{dataset_slice}/"
    else:
        load_path = f"../../datasets/{dataset_name}/"
        
    # load data
    if dataset_name=='DLPFC':
        adata = sc.read_visium(path=load_path, count_file=f"{dataset_slice}_filtered_feature_bc_matrix.h5")
        Ann_df = pd.read_csv(f"{load_path}{dataset_slice}_truth.txt", sep="\t", header=None, index_col=0)
        Ann_df.columns = ["Ground Truth"]
        adata.obs["ground_truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"]
        n_clusters = len(set(adata.obs["ground_truth"].dropna().tolist()))
    if dataset_name=='MB':
        adata = sc.read_visium(path=load_path, count_file=f"{dataset_slice}_filtered_feature_bc_matrix.h5")
        adata.obs["ground_truth"] = None
        n_clusters = None
    elif dataset_name=='CHD':
        adata = sc.read_h5ad(f'{load_path}{dataset_slice}_reset.h5ad')
        adata.obs_names = adata.obs_names + '-1'
        adata.obs["ground_truth"] = adata.obs["region"]
        n_clusters = len(set(adata.obs["ground_truth"].dropna().tolist()))
    elif dataset_name=='BRCA':
        adata = sc.read_visium(path=load_path, count_file=f"{dataset_slice}_filtered_feature_bc_matrix.h5")
        Ann_df = pd.read_csv(f"{load_path}{dataset_slice}_truth.txt", sep="\t", header=None, index_col=0)
        Ann_df.columns = ["Ground Truth"]
        adata.obs["ground_truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"]
        n_clusters = len(set(adata.obs["ground_truth"].dropna().tolist()))
    elif dataset_name=='HER2ST':
        adata = sc.read_h5ad(f'{load_path}{dataset_slice}.h5ad')
        from scipy.sparse import csr_matrix
        adata.X = csr_matrix(adata.X)
        adata.obs["ground_truth"] = adata.obs["annotation"]
        n_clusters = len(set(adata.obs["ground_truth"].dropna().tolist()))-1
    elif dataset_name=='PDAC':
        adata = sc.read_h5ad(f'{load_path}{dataset_slice}.h5ad')
        adata.obs["ground_truth"] = adata.obs["region"]
        n_clusters = len(set(adata.obs["ground_truth"].dropna().tolist()))
        
    return adata, n_clusters