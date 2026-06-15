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


def main(args, num_clusters):
    
    warnings.filterwarnings('ignore')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    seed_all(2023)

    name = args.slice
    # n_clusters = args.n_clusters
        
    ##################################################################################
    result_path = './output/' + name + '/'
    cluster_path = './cluster/' + name + '/DeepGFT/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    ##################################################################################

    # Read data

    path = '../../datasets/' + str(name)  # please replace 'file_fold' with the download path
    adata = sc.read_visium(path, count_file=str(name) + '_filtered_feature_bc_matrix.h5')

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

        silh_mclust =  metrics.silhouette_score(adata.obsm['emb'], adata.obs['mclust'])

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
        adata.obs[cols].to_csv(cluster_path + 'clu_' + str(n_clusters) + '.tsv', sep='\t', index=True)

        silhs = {'n_clusters': str(n_clusters), 'mclust': silh_mclust, 'leiden': silh_leiden, 'louvain': silh_louvain,
                 'kmeans': silh_kmeans}
        results_df = pd.concat([results_df, pd.DataFrame([silhs])], ignore_index=True)

    return results_df



if __name__ == '__main__':
    slice_name = None
    parser = argparse.ArgumentParser(description="Running DeepGFT")
    parser.add_argument('--slice', type=str, default='MBP', help='name of the dataset')
    parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
    args = parser.parse_args()
    slice_name = args.slice

    num_clusters = np.arange(stop=26, start=6, step=1)

    # num_clusters = np.arange(stop=25, start=6, step=1)

    results_df = main(args, num_clusters)

    eva_path = './evaluation/' + slice_name + '/'

    results_df.to_csv(eva_path + "/DeepGFT.csv", index=False)

