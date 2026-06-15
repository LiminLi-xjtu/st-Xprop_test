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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_all(2023)

    name = args.slice
    

    result_path = './output/HER2ST/' + name + '/'
    cluster_path = './cluster/HER2ST/' + name + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    # Read data


    path = f'../../datasets/HER2ST/{name}/'
    adata = sc.read_h5ad(f'{path}{name}.h5ad')
    from scipy.sparse import csr_matrix
    adata.X = csr_matrix(adata.X)

    # y = np.loadtxt(path + '/' + name + "_truth.csv", delimiter=",")
    adata.obs['ground_truth'] = adata.obs['annotation']
    n_clusters = len(np.unique(adata.obs['ground_truth']))-1
    # df = pd.read_csv(path + '/' + name + "_truth.txt", header=None, sep='\s+')
    # adata.obs['domain_names'] = df.iloc[:, 1].values




    # Data preprocessing
    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=3)
    adata, adata_raw = svg(adata, svg_method='gft_top', n_top=3000, ratio_low_freq=5)


    # Build spotnet and genenet
    obtain_spotnet(adata, knn_method='Radius', rad_cutoff=500)
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
    clustering(adata, n_clusters, used_obsm=used_obsm, method='mclust')

    try:
        clustering(adata, n_clusters, used_obsm=used_obsm, method='leiden', start=0.1, end=2.0, increment=0.01)
    except Exception as e:
        adata.obs['leiden'] = None

    try:
        clustering(adata, n_clusters, used_obsm=used_obsm, method='louvain', start=0.1, end=2.0, increment=0.01)
    except Exception as e:
        adata.obs['louvain'] = None

    clustering(adata, n_clusters, used_obsm=used_obsm, method='kmeans')
    
    
    adata2 = adata[adata.obs['ground_truth'] != 'undetermined'].copy()

    ari_mclust = metrics.adjusted_rand_score(adata2.obs['mclust'], adata2.obs['ground_truth'])

    if adata.obs['leiden'][0] is not None:
        ari_leiden = metrics.adjusted_rand_score(adata2.obs['leiden'], adata2.obs['ground_truth'])
    else:
        ari_leiden = None

    if adata.obs['louvain'][0] is not None:
        ari_louvain = metrics.adjusted_rand_score(adata2.obs['louvain'], adata2.obs['ground_truth'])
    else:
        ari_louvain = None

    ari_kmeans = metrics.adjusted_rand_score(adata2.obs['kmeans'], adata2.obs['ground_truth'])

    cols = ['mclust', 'leiden', 'louvain', 'kmeans']
    adata.obs[cols].to_csv(cluster_path + 'DeepGFT.tsv', sep='\t', index=True)

    ############################################################################################################

 

    return ari_mclust, ari_leiden, ari_louvain, ari_kmeans





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Running DeepGFT")
    parser.add_argument('--slice', type=str, default='A1', help='name of the dataset')
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

    # eva_path = './evaluation/HER2ST/' + slice_name + '/'

    # if not os.path.exists(eva_path):
    #     os.makedirs(eva_path)

    # df.to_csv(eva_path + "/DeepGFT.csv", index=False)
    
    import sys
    df.to_csv(sys.stdout, index=False)

