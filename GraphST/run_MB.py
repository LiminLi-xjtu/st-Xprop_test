
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


def main(args, num_clusters):

   # Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
   device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

   dataset = args.slice
   # n_clusters = args.n_clusters


   ##################################################################################
   result_path = '../output/' + dataset + '/'
   cluster_path = '../cluster/' + dataset + '/GraphST/'
   if not os.path.exists(result_path):
      os.makedirs(result_path)
   if not os.path.exists(cluster_path):
      os.makedirs(cluster_path)

   ##################################################################################


   # read data
   file_fold = '../../../datasets/' + str(dataset)  # please replace 'file_fold' with the download path
   adata = sc.read_visium(file_fold, count_file=str(dataset) + '_filtered_feature_bc_matrix.h5', load_images=True) 
   adata.var_names_make_unique()


   # define model
   model = GraphST_model.GraphST(adata, device=device)
   # train model
   adata = model.train()
   emb = adata.obsm['emb']
   np.save(result_path+'GraphST.npy', emb)

   # set radius to specify the number of neighbors considered during refinement
   radius = 50

   columns = ['n_clusters', 'kmeans', 'mclust', 'louvain', 'leiden']
   rows = [str(i) for i in num_clusters]

   results_df = pd.DataFrame(columns=columns)

   for n in range(len(num_clusters)):
      n_clusters = num_clusters[n]

      clustering(adata, n_clusters, radius=radius, method='mclust', refinement=True) # For DLPFC dataset, we use optional refinement step.
   
      try:
         clustering(adata, n_clusters, radius=radius, method='leiden', start=0.1, end=4.0, increment=0.01, refinement=False)
      except Exception as e:
         adata.obs['leiden'] = None
   
      try:
         clustering(adata, n_clusters, radius=radius, method='louvain', start=0.1, end=4.0, increment=0.01, refinement=False)
      except Exception as e:
         adata.obs['louvain'] = None
   
      clustering(adata, n_clusters, radius=radius, method='kmeans', refinement=False)
   
   
      silh_mclust = metrics.silhouette_score(adata.obsm['emb'], adata.obs['mclust'])
   
      if adata.obs['leiden'][0] is not None:
         silh_leiden = metrics.silhouette_score(adata.obsm['emb'], adata.obs['leiden'])
      else:
         silh_leiden = None
   
      if adata.obs['louvain'][0] is not None:
         silh_louvain = metrics.silhouette_score(adata.obsm['emb'], adata.obs['louvain'])
      else:
         silh_louvain = None
   
      silh_kmeans = metrics.silhouette_score(adata.obsm['emb'], adata.obs['kmeans'])
   
      cols = ['mclust', 'leiden', 'louvain', 'kmeans']
      adata.obs[cols].to_csv(cluster_path + 'GraphST.tsv', sep='\t', index=True)


      cols = ['mclust', 'leiden', 'louvain', 'kmeans']
      adata.obs[cols].to_csv(cluster_path + 'clu_' + str(n_clusters) + '.tsv', sep='\t', index=True)

      silhs = {'n_clusters': str(n_clusters), 'mclust': silh_mclust, 'leiden': silh_leiden, 'louvain': silh_louvain,
               'kmeans': silh_kmeans}
      results_df = pd.concat([results_df, pd.DataFrame([silhs])], ignore_index=True)

   return results_df


if __name__ == '__main__':

   slice_name = None
   parser = argparse.ArgumentParser(description="Running GraphST")
   parser.add_argument('--slice', type=str, default='MBC', help='name of the dataset')
   parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
   args = parser.parse_args()
   slice_name = args.slice

   num_clusters = np.arange(stop=26, start=6, step=1)

   # num_clusters = np.arange(stop=25, start=6, step=1)

   results_df = main(args, num_clusters)

   eva_path = '../evaluation/' + slice_name + '/'

   results_df.to_csv(eva_path + "/GraphST.csv", index=False)