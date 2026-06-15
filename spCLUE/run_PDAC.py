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
def main(args):


    sample_name = args.slice
    n_clusters = args.n_clusters

    # sample_name="151671"
    # n_clusters = 5 if sample_name in [str(151669 + x) for x in range(4)] else 7

    ##################################################################################
    result_path = './output/PDAC/' + sample_name + '/'
    cluster_path = './cluster/PDAC/' + sample_name + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    ##################################################################################

    # data_file = f"../../../datasets/DLPFC/{sample_name}.h5ad"
    # adata = sc.read_h5ad(data_file)

    file_fold = '../../datasets/PDAC/' + str(sample_name)  # please replace 'file_fold' with the download path
    # adata = sc.read_visium(file_fold, count_file=str(sample_name) + '_filtered_feature_bc_matrix.h5', load_images=True)

    adata = sc.read_h5ad(file_fold+'.h5ad')

    adata.obs['ground_truth'] = adata.obs['region']
    n_clusters = len(np.unique(adata.obs['ground_truth']))

    adata.var_names_make_unique()
    print(adata)

    # df = pd.read_csv(file_fold + '/' + sample_name + "_truth.txt", header=None, sep='\s+')
    # adata.obs['ground_truth'] = df.iloc[:, 1].values


    # In[3]:


    adata = spCLUE.preprocess(adata)
    adata.obsm["X_pca"] = PCA(n_components=200, random_state=0).fit_transform(adata.X)
    g_spatia = spCLUE.prepare_graph(adata, "spatial")
    g_expr = spCLUE.prepare_graph(adata, "expr")
    graph_dict = {"spatial": g_spatia, "expr":g_expr}


    # ## train spCLUE

    # In[4]:


    spCLUE_model = spCLUE.spCLUE(adata.obsm["X_pca"], graph_dict, n_clusters)
    _, adata.obsm["spCLUE"] = spCLUE_model.train()


    # In[5]:

    emb = adata.obsm["spCLUE"]
    np.save(result_path + 'spCLUE.npy', emb)

    # refinement = True
    # cluster_method = "mclust"
    # pred = spCLUE.clustering(adata, n_clusters, key="spCLUE", refinement=refinement, cluster_methods=cluster_method)

    radius = 50
    used_obsm='spCLUE'

    spCLUE.clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='mclust', start=0.1, end=3.0, increment=0.01, refinement=True)
    try:
        spCLUE.clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='leiden', start=0.1, end=2.0, increment=0.01, refinement=False)
    except Exception as e:
        adata.obs['leiden'] = None

    try:
        spCLUE.clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='louvain', start=0.1, end=2.0, increment=0.01, refinement=False)
    except Exception as e:
        adata.obs['louvain'] = None

    spCLUE.clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='kmeans', refinement=False)

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

    cols = ['mclust', 'leiden', 'louvain', 'kmeans']
    adata.obs[cols].to_csv(cluster_path + 'spCLUE.tsv', sep='\t', index=True)

    ##################################################################################


    return ari_mclust, ari_leiden, ari_louvain, ari_kmeans


############################################################################

if __name__ == '__main__':

   slice_name = None
   parser = argparse.ArgumentParser(description="Running spCLUE")
   parser.add_argument('--slice', type=str, default='st_B', help='name of the dataset')
   parser.add_argument('--n_clusters', type=int, default=4, help='number of clusters')
   args = parser.parse_args()
   slice_name = args.slice

   ari_mclust, ari_leiden, ari_louvain, ari_kmeans = main(args)

   import pandas as pd

   results = {
      'slice': [slice_name],
      'ari_mclust': [ari_mclust],
      'ari_leiden': [ari_leiden],
      'ari_louvain': [ari_louvain],
      'ari_kmeans': [ari_kmeans]
   }
   df = pd.DataFrame(results)
   # print(df.to_csv(index=False))

   eva_path = './evaluation/PDAC/' + slice_name + '/'

   if not os.path.exists(eva_path):
       os.makedirs(eva_path)

   df.to_csv(eva_path + "/spCLUE.csv", index=False)

   # import sys
   # df.to_csv(sys.stdout, index=False)






