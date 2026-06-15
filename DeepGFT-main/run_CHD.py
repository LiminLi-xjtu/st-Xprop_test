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
    # n_clusters = args.n_clusters

    result_path = './output/CHD/' + name + '/'
    cluster_path = './cluster/CHD/' + name + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    # Read data


    path = '../../datasets/CHD/' + name
    # adata = sc.read_visium(path, count_file=name+'_filtered_feature_bc_matrix.h5')


    # adata = sc.read_h5ad(f'{path}/{name}_reset.h5ad')
    adata = sc.read_h5ad(f'{path}/{name}.h5ad')
    adata.obs['ground_truth'] = adata.obs['region']
    n_clusters = len(np.unique(adata.obs['ground_truth']))
    

    # Data preprocessing
    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=3)
    adata, adata_raw = svg(adata, svg_method='gft_top', n_top=3000)


    # Build spotnet and genenet
    # obtain_spotnet(adata, knn_method='Radius', rad_cutoff=150)
    obtain_spotnet(adata, knn_method='KNN', k_cutoff=6)
    
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
    clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='mclust')

    try:
        clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='leiden', start=0.1, end=2.0, increment=0.01)
    except Exception as e:
        adata.obs['leiden'] = None

    try:
        clustering(adata, n_clusters, radius=radius, used_obsm=used_obsm, method='louvain', start=0.1, end=2.0, increment=0.01)
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
    adata.obs[cols].to_csv(cluster_path + 'DeepGFT.tsv', sep='\t', index=True)

    ############################################################################################################


    return ari_mclust, ari_leiden, ari_louvain, ari_kmeans





if __name__ == '__main__':
    slice_name = None
    parser = argparse.ArgumentParser(description="Running DeepGFT")
    parser.add_argument('--slice', type=str, default='D14_measured', help='name of the dataset') #'D14_estimated'
    # parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
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

    df.to_csv(eva_path + "/DeepGFT.csv", index=False)

