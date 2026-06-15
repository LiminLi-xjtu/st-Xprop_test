
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
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   dataset = args.slice
   # n_clusters = args.n_clusters

   result_path = './output/CHD/' + dataset + '/'
   cluster_path = './cluster/CHD/' + dataset + '/'
   if not os.path.exists(result_path):
      os.makedirs(result_path)
   if not os.path.exists(cluster_path):
      os.makedirs(cluster_path)


   # read data
   file_fold = '../../datasets/CHD/' + str(dataset) #please replace 'file_fold' with the download path
   # adata = sc.read_visium(file_fold, count_file=str(dataset)+'_filtered_feature_bc_matrix.h5', load_images=True)

   # path = '../../datasets/CHD/' + name
   # adata = sc.read_visium(path, count_file=name+'_filtered_feature_bc_matrix.h5')

   adata = sc.read_h5ad(f'{file_fold}/{dataset}_reset.h5ad')
   adata.obs['ground_truth'] = adata.obs['region']
   n_clusters = len(np.unique(adata.obs['ground_truth']))

            
   adata.var_names_make_unique()

   # df = pd.read_csv(file_fold + '/' + dataset + "_truth.txt", header=None, sep='\s+')
   # adata.obs['ground_truth'] = df.iloc[:, 1].values

   # define model
   model = GraphST_model.GraphST(adata, device=device, epochs=600, alpha = 1, beta=2)
   # train model
   adata = model.train()
   emb = adata.obsm['emb']
   np.save(result_path+'GraphST.npy', emb)

   # set radius to specify the number of neighbors considered during refinement
   radius = 50

   clustering(adata, n_clusters, method='mclust', refinement=True) # For DLPFC dataset, we use optional refinement step.
   ari_mclust = metrics.adjusted_rand_score(adata.obs['mclust'], adata.obs['ground_truth'])
   
   try:
      clustering(adata, n_clusters, method='leiden', start=0.1, end=2.0, increment=0.01, refinement=False)
   except Exception as e:
      adata.obs['leiden'] = None

   try:
      clustering(adata, n_clusters, method='louvain', start=0.1, end=3.0, increment=0.01, refinement=False)
   except Exception as e:
      adata.obs['louvain'] = None

   clustering(adata, n_clusters, method='kmeans', refinement=False)


   

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
   adata.obs[cols].to_csv(cluster_path + 'GraphST.tsv', sep='\t', index=True)


   return ari_mclust, ari_leiden, ari_louvain, ari_kmeans


if __name__ == '__main__':

   slice_name = None
   parser = argparse.ArgumentParser(description="Running GraphST")
   parser.add_argument('--slice', type=str, default='D10', help='name of the dataset') #'D14_estimated' 'D14_measured'
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

   eva_path = './evaluation/CHD/' + slice_name + '/'

   if not os.path.exists(eva_path):
      os.makedirs(eva_path)

   df.to_csv(eva_path + "/GraphST.csv", index=False)

   # import sys
   # df.to_csv(sys.stdout, index=False)

