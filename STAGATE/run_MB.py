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

def main(args, num_clusters):

    section_id = args.slice
    n_clusters = args.n_clusters

    ##################################################################################
    result_path = '../output/' + section_id + '/'
    cluster_path = '../cluster/' + section_id + '/STAGATE/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    ##################################################################################

    input_dir = '../../../datasets/' + str(section_id)  # please replace 'file_fold' with the download path
    adata = sc.read_visium(input_dir, count_file=str(section_id) + '_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()


    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


    # plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"])


    # ## Constructing the spatial network
    Cal_Spatial_Net(adata, rad_cutoff=150)
    Stats_Spatial_Net(adata)

    # ## Running STAGATE
    # adata = STAGATE.train_STAGATE(adata, alpha=0)
    adata = train_STAGATE(adata, alpha=0)

    emb = adata.obsm['STAGATE']
    np.save(result_path + 'STAGATE.npy', emb)

    ############################################################################################################
    radius = 50
    used_obsm = 'STAGATE'
    columns = ['n_clusters', 'kmeans', 'mclust', 'louvain', 'leiden']
    rows = [str(i) for i in num_clusters]
    results_df = pd.DataFrame(columns=columns)

    for n in range(len(num_clusters)):
        n_clusters = num_clusters[n]

        clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='mclust')

        try:
            clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='leiden', start=0.1, end=3.5,
                       increment=0.01)
        except Exception as e:
            adata.obs['leiden'] = None

        try:
            clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='louvain', start=0.1, end=3.5,
                       increment=0.01)
        except Exception as e:
            adata.obs['louvain'] = None

        clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='kmeans')

        silh_mclust = metrics.silhouette_score(adata.obsm[used_obsm], adata.obs['mclust'])

        if adata.obs['leiden'][0] is not None:
            silh_leiden = metrics.silhouette_score(adata.obsm[used_obsm], adata.obs['leiden'])
        else:
            silh_leiden = None

        if adata.obs['louvain'][0] is not None:
            silh_louvain = metrics.silhouette_score(adata.obsm[used_obsm], adata.obs['louvain'])
        else:
            silh_louvain = None

        silh_kmeans = metrics.silhouette_score(adata.obsm[used_obsm], adata.obs['kmeans'])

        cols = ['mclust', 'leiden', 'louvain', 'kmeans']
        adata.obs[cols].to_csv(cluster_path + 'clu_' + str(n_clusters) + '.tsv', sep='\t', index=True)

        silhs = {'n_clusters': str(n_clusters), 'mclust': silh_mclust, 'leiden': silh_leiden, 'louvain': silh_louvain,
                 'kmeans': silh_kmeans}
        results_df = pd.concat([results_df, pd.DataFrame([silhs])], ignore_index=True)

    return results_df


if __name__ == '__main__':
    slice_name = None
    parser = argparse.ArgumentParser(description="Running STAGATE")
    parser.add_argument('--slice', type=str, default='MBP', help='name of the dataset')
    parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
    args = parser.parse_args()
    slice_name = args.slice

    num_clusters = np.arange(stop=26, start=6, step=1)

    # num_clusters = np.arange(stop=25, start=6, step=1)

    results_df = main(args, num_clusters)

    eva_path = '../evaluation/' + slice_name + '/'

    results_df.to_csv(eva_path + "/STAGATE.csv", index=False)