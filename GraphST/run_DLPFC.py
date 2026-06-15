
import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
import numpy as np
import argparse
from sklearn.cluster import KMeans

import GraphST_model
from utils import clustering


def main(args):

   # Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
   device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

   dataset = args.slice
   n_clusters = args.n_clusters

   result_path = './output/DLPFC/' + dataset + '/'
   cluster_path = './cluster/DLPFC/' + dataset + '/'
   if not os.path.exists(result_path):
      os.makedirs(result_path)
   if not os.path.exists(cluster_path):
      os.makedirs(cluster_path)


   # read data
   file_fold = '../../datasets/DLPFC/' + str(dataset) #please replace 'file_fold' with the download path
   adata = sc.read_visium(file_fold, count_file=str(dataset)+'_filtered_feature_bc_matrix.h5', load_images=True)
   adata.var_names_make_unique()


   df_label = pd.read_csv(file_fold+'/'+dataset+"_truth.txt", sep='\t', header=None, index_col=0)
   df_label.columns = ['layer_guess']
   adata.obs['ground_truth'] =df_label['layer_guess']

   # define model
   model = GraphST_model.GraphST(adata, device=device)
   # train model
   adata = model.train()
   emb = adata.obsm['emb']
   np.save(result_path+'GraphST.npy', emb)

   # set radius to specify the number of neighbors considered during refinement
   radius = 50

   clustering(adata, n_clusters, radius=radius, method='mclust', refinement=True) # For DLPFC dataset, we use optional refinement step.

   try:
      clustering(adata, n_clusters, radius=radius, method='leiden', start=0.1, end=2.0, increment=0.01, refinement=False)
   except Exception as e:
      adata.obs['leiden'] = None

   try:
      clustering(adata, n_clusters, radius=radius, method='louvain', start=0.1, end=2.0, increment=0.01, refinement=False)
   except Exception as e:
      adata.obs['louvain'] = None

   clustering(adata, n_clusters, radius=radius, method='kmeans', refinement=False)


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
   adata.obs[cols].to_csv(cluster_path + 'GraphST_na.tsv', sep='\t', index=True)


   # filter out NA nodes
   df = pd.read_csv(file_fold + '/' + dataset + "_truth.txt", header=None, sep='\s+')
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

   adata.obs[cols].to_csv(cluster_path + 'GraphST.tsv', sep='\t', index=True)

   return ari_mclust_na, ari_leiden_na, ari_louvain_na, ari_kmeans_na, ari_mclust, ari_leiden, ari_louvain, ari_kmeans


if __name__ == '__main__':

   slice_name = None
   parser = argparse.ArgumentParser(description="Running GraphST")
   parser.add_argument('--slice', type=str, default='151675', help='name of the dataset')
   parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
   args = parser.parse_args()
   slice_name = args.slice

   ari_mclust_na, ari_leiden_na, ari_louvain_na, ari_kmeans_na, ari_mclust, ari_leiden, ari_louvain, ari_kmeans = main(args)

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

