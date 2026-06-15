#!/usr/bin/env python
# coding: utf-8

# ## Loading packages

# In[1]:


import scanpy as sc 
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

import scipy.sparse as sp 
import warnings
import os

from sklearn import metrics
import multiprocessing as mp
import numpy as np
import argparse


warnings.filterwarnings("ignore")

# os.environ["R_HOME"] = "/home/lxx/.conda/envs/r4Base/lib/R"
# os.environ["R_USER"] = "/home/lxx/.local/lib/python3.9/site-packages/rpy2"

import spCLUE

spCLUE.fix_seed(0)


# ## Loading data

# In[2]:
def main(args, num_clusters):


    sample_name = args.slice


    # sample_name="151671"
    # n_clusters = 5 if sample_name in [str(151669 + x) for x in range(4)] else 7

    ##################################################################################
    result_path = '../output/' + sample_name + '/spCLUE/'
    cluster_path = '../cluster/' + sample_name + '/spCLUE/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    ##################################################################################

    # data_file = f"../../../datasets/DLPFC/{sample_name}.h5ad"
    # adata = sc.read_h5ad(data_file)

    file_fold = '../../../datasets/' + str(sample_name)  # please replace 'file_fold' with the download path
    adata = sc.read_visium(file_fold, count_file=str(sample_name) + '_filtered_feature_bc_matrix.h5', load_images=True)

    adata.var_names_make_unique()
    print(adata)


    adata = spCLUE.preprocess(adata)
    adata.obsm["X_pca"] = PCA(n_components=200, random_state=0).fit_transform(adata.X)
    g_spatia = spCLUE.prepare_graph(adata, "spatial")
    g_expr = spCLUE.prepare_graph(adata, "expr")
    graph_dict = {"spatial": g_spatia, "expr":g_expr}


    # ## train spCLUE

    radius = 50
    used_obsm='spCLUE'

    columns = ['n_clusters', 'kmeans', 'mclust', 'louvain', 'leiden']
    rows = [str(i) for i in num_clusters]

    results_df = pd.DataFrame(columns=columns)

    for n in range(len(num_clusters)):
        n_clusters = num_clusters[n]

        spCLUE_model = spCLUE.spCLUE(adata.obsm["X_pca"], graph_dict, n_clusters)
        _, adata.obsm["spCLUE"] = spCLUE_model.train()
        emb = adata.obsm["spCLUE"]
        np.save(result_path + 'emb_' + str(n_clusters)+ '.npy', emb)

        spCLUE.clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='mclust', start=0.05, end=3.0, increment=0.01, refinement=True)
        try:
            spCLUE.clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='leiden', start=0.05, end=2.0, increment=0.01, refinement=False)
        except Exception as e:
            adata.obs['leiden'] = None

        try:
            spCLUE.clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='louvain', start=0.1, end=3.0, increment=0.01, refinement=False)
        except Exception as e:
            adata.obs['louvain'] = None

        spCLUE.clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='kmeans', refinement=False)

        silh_mclust = metrics.silhouette_score(adata.obsm["spCLUE"], adata.obs['mclust'])

        if adata.obs['leiden'][0] is not None:
            silh_leiden = metrics.silhouette_score(adata.obsm["spCLUE"], adata.obs['leiden'])
        else:
            silh_leiden = None

        if adata.obs['louvain'][0] is not None:
            silh_louvain = metrics.silhouette_score(adata.obsm["spCLUE"], adata.obs['louvain'])
        else:
            silh_louvain = None

        silh_kmeans = metrics.silhouette_score(adata.obsm["spCLUE"], adata.obs['kmeans'])

        cols = ['mclust', 'leiden', 'louvain', 'kmeans']
        adata.obs[cols].to_csv(cluster_path + 'clu_' + str(n_clusters) + '.tsv', sep='\t', index=True)


        silhs = {'n_clusters': str(n_clusters), 'mclust': silh_mclust, 'leiden': silh_leiden, 'louvain': silh_louvain, 'kmeans': silh_kmeans}
        results_df = pd.concat([results_df, pd.DataFrame([silhs])], ignore_index=True)


    return results_df


############################################################################

if __name__ == '__main__':

   slice_name = None
   parser = argparse.ArgumentParser(description="Running spCLUE")
   parser.add_argument('--slice', type=str, default='MBP', help='name of the dataset')
   parser.add_argument('--n_clusters', type=int, default=5, help='number of clusters')
   args = parser.parse_args()
   slice_name = args.slice
   num_clusters = np.arange(stop=26, start=6, step=1)

   # num_clusters = np.arange(stop=25, start=6, step=1)

   results_df = main(args, num_clusters)

   eva_path = '../evaluation/' + slice_name + '/'

   results_df.to_csv(eva_path + "/spCLUE.csv", index=False)







