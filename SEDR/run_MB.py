import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import argparse

import os
from pathlib import Path
import warnings
import numpy as np
import SEDR
from SEDR.utils import clustering
# from SEDR.graph_func import graph_construction, combine_graph_dict
# from SEDR.utils_func import adata_preprocess, fix_seed
# from SEDR.SEDR_model import Sedr
# from SEDR.clustering_func import  mclust_R, leiden, louvain

warnings.filterwarnings('ignore')

def main(args, num_clusters):

    random_seed = 2023
    SEDR.fix_seed(random_seed)
    # gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # path
    # data_root = '../../../datasets/DLPFC'

    # sample name
    sample_name = args.slice
    # n_clusters = args.n_clusters

    ##################################################################################
    result_path = '../output/' + sample_name + '/'
    cluster_path = '../cluster/' + sample_name + '/SEDR/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    ##################################################################################

    # adata = sc.read_visium(data_root / sample_name)

    data_root = '../../../datasets/' + str(sample_name)  # please replace 'file_fold' with the download path
    adata = sc.read_visium(data_root, count_file=str(sample_name) + '_filtered_feature_bc_matrix.h5')

    adata.var_names_make_unique()


    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
    adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X



    graph_dict = SEDR.graph_construction(adata, 12)
    print(graph_dict)

    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
    using_dec = True
    if using_dec:
        sedr_net.train_with_dec(N=1)
    else:
        sedr_net.train_without_dec(N=1)
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['SEDR'] = sedr_feat


    np.save(result_path + 'SEDR.npy', sedr_feat)


    # SEDR.mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR')

    ############################################################################################################
    radius = 50
    used_obsm = 'SEDR'
    columns = ['n_clusters', 'kmeans', 'mclust', 'louvain', 'leiden']
    rows = [str(i) for i in num_clusters]
    results_df = pd.DataFrame(columns=columns)

    for n in range(len(num_clusters)):
        n_clusters = num_clusters[n]

        clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='mclust')

        try:
            clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='leiden', start=0.1, end=3.0,
                       increment=0.01)
        except Exception as e:
            adata.obs['leiden'] = None

        try:
            clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='louvain', start=0.1, end=3.0,
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
    parser = argparse.ArgumentParser(description="Running SEDR")
    parser.add_argument('--slice', type=str, default='MBP', help='name of the dataset')
    parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
    args = parser.parse_args()
    slice_name = args.slice

    num_clusters = np.arange(stop=26, start=6, step=1)

    # num_clusters = np.arange(stop=25, start=6, step=1)

    results_df = main(args, num_clusters)

    eva_path = '../evaluation/' + slice_name + '/'

    results_df.to_csv(eva_path + "/SEDR.csv", index=False)