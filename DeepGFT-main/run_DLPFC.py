from matplotlib import pyplot as plt
from DeepGFT.utils import *
from DeepGFT.genenet import obtain_genenet
from DeepGFT.train import *
import torch
import scanpy as sc
import warnings
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
from DeepGFT.utils import clustering

# os.environ['R_HOME'] = '/users/PCON0022/jxliu/anaconda3/envs/DeepGFT/lib/R'


def main(args):
    warnings.filterwarnings('ignore')
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    seed_all(2023)

    name = args.slice
    n_clusters = args.n_clusters

    result_path = './output/DLPFC/' + name + '/'
    cluster_path = './cluster/DLPFC/' + name + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    # Read data


    path = '../../datasets/DLPFC/' + name
    adata = sc.read_visium(path, count_file=name+'_filtered_feature_bc_matrix.h5')

    # y = np.loadtxt(path + '/' + name + "_truth.csv", delimiter=",")
    # adata.obs['ground_truth'] = y

    df = pd.read_csv(path + '/' + name + "_truth.txt", header=None, sep='\s+')
    adata.obs['domain_names'] = df.iloc[:, 1].values




    # Data preprocessing
    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=3)
    adata, adata_raw = svg(adata, svg_method='gft_top', n_top=3000)


    # Build spotnet and genenet
    obtain_spotnet(adata, knn_method='Radius', rad_cutoff=150)
    gene_freq_mtx, gene_eigvecs_T, gene_eigvals = f2s_gene(adata, gene_signal=1500, c1=0.05)
    obtain_genenet(adata)
    spot_freq_mtx, spot_eigvecs_T, spot_eigvals = f2s_spot(adata, spot_signal=1500, middle=3, c2=0.005)
    obtain_pre_spotnet(adata, adata_raw)


    res, lamda, emb_spot, _, attention = train_spot(adata, gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T,
                                                    alpha=20, device=device, epoch_max=600)


    adata.obsm['emb'] = emb_spot


    np.save(result_path + 'DeepGFT.npy', emb_spot)

    ############################################################################################################
    radius = 50
    used_obsm = 'emb'
    clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='mclust', refinement=True)

    try:
        clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='leiden', start=0.1, end=2.0, increment=0.01, refinement=True)
    except Exception as e:
        adata.obs['leiden'] = None

    try:
        clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='louvain', start=0.1, end=2.0, increment=0.01, refinement=True)
    except Exception as e:
        adata.obs['louvain'] = None

    clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='kmeans', refinement=True)

    # ari_mclust_na = metrics.adjusted_rand_score(adata.obs['mclust'], adata.obs['ground_truth'])

    # if adata.obs['leiden'][0] is not None:
    #     ari_leiden_na = metrics.adjusted_rand_score(adata.obs['leiden'], adata.obs['ground_truth'])
    # else:
    #     ari_leiden_na = None

    # if adata.obs['louvain'][0] is not None:
    #     ari_louvain_na = metrics.adjusted_rand_score(adata.obs['louvain'], adata.obs['ground_truth'])
    # else:
    #     ari_louvain_na = None

    # ari_kmeans_na = metrics.adjusted_rand_score(adata.obs['kmeans'], adata.obs['ground_truth'])

    cols = ['mclust', 'leiden', 'louvain', 'kmeans']
    # adata.obs[cols].to_csv(cluster_path + 'DeepGFT_na.tsv', sep='\t', index=True)

    ############################################################################################################

    # filter out NA nodes
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

    adata.obs[cols].to_csv(cluster_path + 'DeepGFT.tsv', sep='\t', index=True)

    # print('Success!')

    return ari_mclust, ari_leiden, ari_louvain, ari_kmeans #ari_mclust_na, ari_leiden_na, ari_louvain_na, ari_kmeans_na, 





if __name__ == '__main__':
    slice_name = None
    parser = argparse.ArgumentParser(description="Running DeepGFT")
    parser.add_argument('--slice', type=str, default='151507', help='name of the dataset')
    parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
    args = parser.parse_args()
    slice_name = args.slice

    ari_mclust, ari_leiden, ari_louvain, ari_kmeans = main(args) #ari_mclust_na, ari_leiden_na, ari_louvain_na, ari_kmeans_na, 

    import pandas as pd

    results = {
        'slice': [slice_name],
        # 'ari_mclust_na': [ari_mclust_na],
        # 'ari_leiden_na': [ari_leiden_na],
        # 'ari_louvain_na': [ari_louvain_na],
        # 'ari_kmeans_na': [ari_kmeans_na],
        'ari_mclust': [ari_mclust],
        'ari_leiden': [ari_leiden],
        'ari_louvain': [ari_louvain],
        'ari_kmeans': [ari_kmeans]
    }
    df = pd.DataFrame(results)

    
    
    # import os
    # output_file="../evaluation/DLPFC/DeepGFT.csv"
    # write_header = not os.path.exists(output_file)

    # df.to_csv(
    #     output_file,
    #     mode='a',
    #     header=write_header,
    #     index=False
    # )


    import sys

    df.to_csv(sys.stdout, index=False)

