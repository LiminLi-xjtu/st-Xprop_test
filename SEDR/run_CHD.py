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

def main(args):

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
    result_path = './output/CHD/' + sample_name + '/'
    cluster_path = './cluster/CHD/' + sample_name + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    ##################################################################################

    # adata = sc.read_visium(data_root / sample_name)

    data_root = '../../datasets/CHD/' + sample_name
    # adata = sc.read_visium(data_root, count_file=sample_name + '_filtered_feature_bc_matrix.h5')
    # df = pd.read_csv(data_root + '/' + sample_name + "_truth.txt", header=None, sep='\s+')
    # adata.obs['ground_truth'] = df.iloc[:, 1].values

    adata = sc.read_h5ad(f'{data_root}/{sample_name}_reset.h5ad')
    adata.obs['ground_truth'] = adata.obs['region']
    n_clusters = len(np.unique(adata.obs['ground_truth']))


    adata.var_names_make_unique()

    # df_meta = pd.read_csv(data_root / sample_name / 'metadata.tsv', sep='\t')
    # adata.obs['layer_guess'] = df_meta['layer_guess']




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
    clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='mclust')

    try:
        clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='leiden', start=0.1, end=2.0,
                   increment=0.01)
    except Exception as e:
        adata.obs['leiden'] = None

    try:
        clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='louvain', start=0.1, end=3.0,
                   increment=0.01)
    except Exception as e:
        adata.obs['louvain'] = None

    clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='kmeans')

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
    adata.obs[cols].to_csv(cluster_path + 'SEDR.tsv', sep='\t', index=True)

    ############################################################################################################


    return ari_mclust, ari_leiden, ari_louvain, ari_kmeans


if __name__ == '__main__':
    slice_name = None
    parser = argparse.ArgumentParser(description="Running SEDR")
    parser.add_argument('--slice', type=str, default='D14_estimated', help='name of the dataset') #'D14_estimated' 'D14_measured'
    # parser.add_argument('--n_clusters', type=int, default=20, help='number of clusters')
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

    df.to_csv(eva_path + "/SEDR.csv", index=False)

    # import sys
    #
    # df.to_csv(sys.stdout, index=False)

