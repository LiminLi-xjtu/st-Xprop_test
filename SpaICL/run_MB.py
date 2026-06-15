#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import torch
import scanpy as sc
import pandas as pd
import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score as ari_score
from utils.Func import *
from utils.Utils import *
from SpaICL import spaicl

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.decomposition import PCA
import argparse
from sklearn import metrics

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2023):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    robjects.r.library("mclust")

    # rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    # res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    # mclust_res = np.array(res[-2])

    with localconverter(robjects.default_converter + numpy2ri.converter):
        r_data = robjects.conversion.py2rpy(adata.obsm[used_obsm])
        res = rmclust(r_data, num_cluster, modelNames)
    # mclust_res = np.array(res.rx2('classification'))
    classification_idx = list(res.names()).index('classification')
    mclust_res = np.array(res[classification_idx])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def load_embeddings_from_h5(h5_path):
    import h5py
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
        barcodes = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f['barcodes'][:]]
    return barcodes, features


def off_the_shelf_adata(data_name, slice):
    # adata = sc.read_h5ad(f'data/{slice}.h5ad')
    # adata.obsm['img_pca'] = torch.tensor(adata.obsm['img_pca'])
    
    path = f'../../datasets/{data_name}/{slice}' 
    adata = sc.read_visium(path, count_file=f'{slice}_filtered_feature_bc_matrix.h5')
    
    df_label = pd.read_csv(f"{path}/{slice}_truth.txt", sep='\t', header=None, index_col=0)
    df_label.columns = ['layer_guess']
    adata.obs['layer_guess'] = df_label['layer_guess']
    
    h5_path = f"{path}/vit_patch_concat_embeddings.h5"
    data_image_h5 = load_embeddings_from_h5(h5_path)
    barcode_to_index = {barcode: i for i, barcode in enumerate(data_image_h5[0])}
    common_barcodes = [bc for bc in adata.obs_names if bc in barcode_to_index]
    reordered_indices = [barcode_to_index[bc] for bc in common_barcodes]
    data_image = data_image_h5[1][reordered_indices, :]
    adata = adata[np.array(common_barcodes), :]
    
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=config['data']['top_genes'])
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X
    
    image_feat_pca = PCA(n_components=200, random_state=42).fit_transform(data_image)
    adata.obsm['img_pca'] = torch.tensor(image_feat_pca)

    return adata


# In[4]:

def main(args, num_clusters):
    
    with open('./SpaICL-main/config/MBA_CLFS.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    proj_name = args.slice
    # n_clusters = args.n_clusters
        
    ##################################################################################
    result_path = './output/' + proj_name + '/SpaICL/'
    cluster_path = './cluster/' + proj_name + '/SpaICL/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    ##################################################################################

    # Read data

    path = '../../datasets/MB/' + str(proj_name)  # please replace 'file_fold' with the download path
    adata = sc.read_visium(path, count_file=str(proj_name) + '_filtered_feature_bc_matrix.h5')
    
    adata.var_names_make_unique()

    h5_path = f"{path}/vit_patch_concat_embeddings.h5"
    data_image_h5 = load_embeddings_from_h5(h5_path)
    barcode_to_index = {barcode: i for i, barcode in enumerate(data_image_h5[0])}
    common_barcodes = [bc for bc in adata.obs_names if bc in barcode_to_index]
    reordered_indices = [barcode_to_index[bc] for bc in common_barcodes]
    data_image = data_image_h5[1][reordered_indices, :]
    adata = adata[np.array(common_barcodes), :]
    
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=config['data']['top_genes'])
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X
    
    image_feat_pca = PCA(n_components=200, random_state=42).fit_transform(data_image)
    adata.obsm['img_pca'] = torch.tensor(image_feat_pca)

    graph_dict = graph_construction(adata, config['data']['k_cutoff'])
    
    columns = ['n_clusters', 'mclust']
    rows = [str(i) for i in num_clusters]
    results_df = pd.DataFrame(columns=columns)
    
    for n in range(len(num_clusters)):
        n_clusters = num_clusters[n]

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        net = spaicl(adata, graph_dict=graph_dict, num_clusters=n_clusters, device=device, config=config)
        para_list = net.train()

        enc_rep, recon = net.process()
        enc_rep = enc_rep.data.cpu().numpy()
        recon = recon.data.cpu().numpy()
        adata.obsm['latent'] = enc_rep
        adata.obsm['recon'] = recon


        # In[9]:
        
        np.save(result_path + 'clu_' + str(n_clusters) +'.npy', adata.obsm['latent'])
        
        mclust_R(adata, num_cluster=n_clusters, used_obsm='latent')

        silh_mclust =  metrics.silhouette_score(adata.obsm['latent'], adata.obs['mclust'])

        cols = ['mclust']
        adata.obs[cols].to_csv(cluster_path + 'clu_' + str(n_clusters) + '.tsv', sep='\t', index=True)

        silhs = {'n_clusters': str(n_clusters), 'mclust': silh_mclust}
        results_df = pd.concat([results_df, pd.DataFrame([silhs])], ignore_index=True)

    return results_df



if __name__ == '__main__':
    slice_name = None
    parser = argparse.ArgumentParser(description="Running SpaICL")
    parser.add_argument('--slice', type=str, default='MBP', help='name of the dataset')
    parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
    args = parser.parse_args()
    slice_name = args.slice

    num_clusters = np.arange(stop=26, start=6, step=1)

    # num_clusters = np.arange(stop=25, start=6, step=1)

    results_df = main(args, num_clusters)

    eva_path = './evaluation/' + slice_name + '/'

    results_df.to_csv(eva_path + "/SpaICL.csv", index=False)

