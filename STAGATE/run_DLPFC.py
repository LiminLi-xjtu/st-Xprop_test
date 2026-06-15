#!/usr/bin/env python
# coding: utf-8

# # Tutorial 1: 10x Visium (DLPFC dataset)

# Here we present our re-analysis of 151676 sample of the dorsolateral prefrontal cortex (DLPFC) dataset. Maynard et al. has manually annotated DLPFC layers and white matter (WM) based on the morphological features and gene markers.
# 
# This tutorial demonstrates how to identify spatial domains on 10x Visium data using STAGATE. The processed data are available at https://github.com/LieberInstitute/spatialLIBD. We downloaded the manual annotation from the spatialLIBD package and provided at https://drive.google.com/drive/folders/10lhz5VY7YfvHrtV40MwaqLmWz56U9eBP?usp=sharing.

# ## Preparation

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import argparse
from sklearn import metrics

# In[3]:


from sklearn.metrics.cluster import adjusted_rand_score


# In[4]:


# import STAGATE
from utils import *
from Train_STAGATE import train_STAGATE
# In[5]:


# the location of R (used for the mclust clustering)
# os.environ['R_HOME'] = 'D:\Program Files\R\R-4.0.3'
# os.environ['R_USER'] = 'D:\ProgramData\Anaconda3\Lib\site-packages\rpy2'


# In[6]:

def main(args):

    section_id = args.slice
    n_clusters = args.n_clusters

    result_path = '../output/DLPFC/' + section_id + '/'
    cluster_path = '../cluster/DLPFC/' + section_id + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    input_dir = '../../../datasets/DLPFC/' + section_id
    adata = sc.read_visium(path=input_dir, count_file=section_id+'_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    y = np.loadtxt(input_dir+'/'+section_id + "_truth.csv", delimiter=",")
    adata.obs['ground_truth'] = y


    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)



    # read the annotation
    # Ann_df = np.loadtxt(input_dir+'/'+section_id+"_truth.csv",delimiter=",")
    # Ann_df = pd.read_csv(input_dir+'/'+section_id + "_truth.txt", header=None, sep='\s+', index_col=0)
    # Ann_df = pd.read_csv(os.path.join('Data', section_id, section_id+'_truth.txt'), sep='\t', header=None, index_col=0)
    # Ann_df.columns = ['Ground Truth']



    # adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']




    # plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"])


    # ## Constructing the spatial network



    # STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
    # STAGATE.Stats_Spatial_Net(adata)
    Cal_Spatial_Net(adata, rad_cutoff=150)
    Stats_Spatial_Net(adata)

    # ## Running STAGATE
    # adata = STAGATE.train_STAGATE(adata, alpha=0)
    adata = train_STAGATE(adata, alpha=0)

    emb = adata.obsm['STAGATE']
    np.save(result_path + 'STAGATE.npy', emb)

    ############################################################################################################
    radius = 50
    clustering(adata, n_clusters, radius=radius, method='mclust')

    try:
        clustering(adata, n_clusters, radius=radius, method='leiden', start=0.1, end=2.0, increment=0.01)
    except Exception as e:
        adata.obs['leiden'] = None

    try:
        clustering(adata, n_clusters, radius=radius, method='louvain', start=0.1, end=2.0, increment=0.01)
    except Exception as e:
        adata.obs['louvain'] = None

    clustering(adata, n_clusters, radius=radius, method='kmeans')

    ari_mclust_na = metrics.adjusted_rand_score(adata.obs['mclust'], adata.obs['ground_truth'])

    if adata.obs['leiden'][0] is not None:
        ari_leiden_na = metrics.adjusted_rand_score(adata.obs['leiden'], adata.obs['ground_truth'])
    else:
        ari_leiden_na = None

    if adata.obs['louvain'][0] is not None:
        ari_louvain_na = metrics.adjusted_rand_score(adata.obs['louvain'], adata.obs['ground_truth'])
    else:
        ari_louvain_na = None

    ari_kmeans_na = metrics.adjusted_rand_score(adata.obs['kmeans'], adata.obs['ground_truth'])

    cols = ['mclust', 'leiden', 'louvain', 'kmeans']
    adata.obs[cols].to_csv(cluster_path + 'STAGATE_na.tsv', sep='\t', index=True)

    ############################################################################################################

    # filter out NA nodes
    df = pd.read_csv(input_dir+'/'+section_id + "_truth.txt", header=None, sep='\s+')
    adata.obs['domain_names'] = df.iloc[:, 1].values
    adata = adata[~pd.isnull(adata.obs['domain_names'])]

    # calculate metric ARI
    ari_mclust = metrics.adjusted_rand_score(adata.obs['mclust'], adata.obs['ground_truth'])

    if adata.obs['leiden'][0] is not None:
        ari_leiden = metrics.adjusted_rand_score(adata.obs['leiden'], adata.obs['ground_truth'])
    else:
        ari_leiden = None

    if adata.obs['louvain'][0] is not None:
        ari_louvain = metrics.adjusted_rand_score(adata.obs['louvain'], adata.obs['ground_truth'])
    else:
        ari_louvain = None

    ari_kmeans = metrics.adjusted_rand_score(adata.obs['kmeans'], adata.obs['ground_truth'])

    adata.obs[cols].to_csv(cluster_path + 'STAGATE.tsv', sep='\t', index=True)

    return ari_mclust_na, ari_leiden_na, ari_louvain_na, ari_kmeans_na, ari_mclust, ari_leiden, ari_louvain, ari_kmeans

    # sc.pp.neighbors(adata, use_rep='STAGATE')
    # sc.tl.umap(adata)
    # adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=7)
    # adata = mclust_R(adata, used_obsm='STAGATE', num_cluster=7)

    # In[16]:


    # obs_df = adata.obs.dropna()
    # ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
    # print('Adjusted rand index = %.2f' %ARI)


    # In[17]:


    # plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.umap(adata, color=["mclust", "Ground Truth"], title=['STAGATE (ARI=%.2f)'%ARI, "Ground Truth"])


    # In[18]:


    # plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.spatial(adata, color=["mclust", "Ground Truth"], title=['STAGATE (ARI=%.2f)'%ARI, "Ground Truth"])


    # ## Spatial trajectory inference (PAGA)

    # In[19]:


    # used_adata = adata[adata.obs['Ground Truth']!='nan',]
    # used_adata
    #
    #
    # # In[20]:
    #
    #
    # sc.tl.paga(used_adata, groups='Ground Truth')
    #
    #
    # # In[21]:
    #
    #
    # plt.rcParams["figure.figsize"] = (4,3)
    # sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
    #                    title=section_id+'_STGATE', legend_fontoutline=2, show=False)
    #

if __name__ == '__main__':
    slice_name = None
    parser = argparse.ArgumentParser(description="Running STAGATE")
    parser.add_argument('--slice', type=str, default='151675', help='name of the dataset')
    parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
    args = parser.parse_args()
    slice_name = args.slice

    ari_mclust_na, ari_leiden_na, ari_louvain_na, ari_kmeans_na, ari_mclust, ari_leiden, ari_louvain, ari_kmeans = main(
        args)

    import pandas as pd

    results = {
        'slice': [slice_name],
        'ari_mclust_na': [ari_mclust_na],
        'ari_leiden_na': [ari_leiden_na],
        'ari_louvain_na': [ari_louvain_na],
        'ari_kmeans_na': [ari_kmeans_na],
        'ari_mclust': [ari_mclust],
        'ari_leiden': [ari_leiden],
        'ari_louvain': [ari_louvain],
        'ari_kmeans': [ari_kmeans]
    }
    df = pd.DataFrame(results)
    # print(df.to_csv(index=False))

    import sys

    df.to_csv(sys.stdout, index=False)

