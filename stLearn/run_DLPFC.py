import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
                            homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scanpy as sc
import stlearn as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import sys
import matplotlib.pyplot as plt

from sklearn import metrics
import multiprocessing as mp
import numpy as np
import argparse
from sklearn.cluster import KMeans

from utils import clustering

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def calculate_clustering_matrix(pred, gt, sample, methods_):
    df = pd.DataFrame(columns=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"])

    pca_ari = adjusted_rand_score(pred, gt)
    df = df.append(pd.Series([sample, pca_ari, "pca", methods_, "Adjusted_Rand_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    pca_nmi = normalized_mutual_info_score(pred, gt)
    df = df.append(pd.Series([sample, pca_nmi, "pca", methods_, "Normalized_Mutual_Info_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    pca_purity = purity_score(pred, gt)
    df = df.append(pd.Series([sample, pca_purity, "pca", methods_, "Purity_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    pca_homogeneity, pca_completeness, pca_v_measure = homogeneity_completeness_v_measure(pred, gt)

    df = df.append(pd.Series([sample, pca_homogeneity, "pca", methods_, "Homogeneity_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)


    df = df.append(pd.Series([sample, pca_completeness, "pca", methods_, "Completeness_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

    df = df.append(pd.Series([sample, pca_v_measure, "pca", methods_, "V_Measure_Score"],
                             index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)
    return df

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def main(args):


    sample = args.slice
    n_clusters = args.n_clusters

    BASE_PATH = Path('../../datasets/DLPFC/'+sample)
    file_fold = '../../datasets/DLPFC/' + sample

    # dir_input = f'../data/DLPFC/{sample}/'
    # dir_output = f'../output/{sample}/stLearn/'
    #
    # if not os.path.exists(dir_output):
    #     os.makedirs(dir_output)
    
    
    ##################################################################################
    result_path = './output/DLPFC/' + sample + '/'
    cluster_path = './cluster/DLPFC/' + sample + '/'
    if not os.path.exists(result_path):
      os.makedirs(result_path)
    if not os.path.exists(cluster_path):
      os.makedirs(cluster_path)
    
    ##################################################################################
    
    # n_clusters=7
    clu=str(n_clusters)
    TILE_PATH = Path("./tmp/{}_tiles".format(sample))
    TILE_PATH.mkdir(parents=True, exist_ok=True)
    
    # OUTPUT_PATH = Path(f"../output/DLPFC/{sample}/stLearn")
    # OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    data = st.Read10X(BASE_PATH, count_file=str(sample)+'_filtered_feature_bc_matrix.h5')
    data  = st.convert_scanpy(data)
    
    y = np.loadtxt(file_fold + '/' + sample + "_truth.csv", delimiter=",")
    data.obs['ground_truth'] = y
    
    le = LabelEncoder()

    # pre-processing for gene count table
    st.pp.filter_genes(data,min_cells=1)
    st.pp.normalize_total(data)
    st.pp.log1p(data)


    # pre-processing for spot image
    st.pp.tiling(data, TILE_PATH)
    # this step uses deep learning model to extract high-level features from tile images
    # may need few minutes to be completed
    st.pp.extract_feature(data)
    
    
    # run PCA for gene expression data
    st.em.run_pca(data,n_comps=50)    
    
    # stSME
    st.spatial.SME.SME_normalize(data, use_data="raw") #, weights="physical_distance"
    data_ = data.copy()
    data_.X = data_.obsm['raw_SME_normalized']
    st.pp.scale(data_)
    st.em.run_pca(data_,n_comps=30)
    
    emb = data_.obsm['X_pca']
    np.save(result_path + 'stLearn.npy', emb)
    
    ##################################################################################################
    
    radius = 50
    
    clustering(data_, n_clusters, method='mclust',
               refinement=True)  # For DLPFC dataset, we use optional refinement step.
    
    try:
        clustering(data_, n_clusters, method='leiden', start=0.1, end=2.0, increment=0.01, refinement=False)
    except Exception as e:
        data_.obs['leiden'] = None
    
    try:
        clustering(data_, n_clusters, method='louvain', start=0.1, end=2.0, increment=0.01, refinement=False)
    except Exception as e:
        data_.obs['louvain'] = None
    
    clustering(data_, n_clusters, method='kmeans', refinement=False)
    
    ari_mclust_na = metrics.adjusted_rand_score(data_.obs['mclust'], data_.obs['ground_truth'])
    
    if data_.obs['leiden'][0] is not None:
        ari_leiden_na = metrics.adjusted_rand_score(data_.obs['leiden'], data_.obs['ground_truth'])
    else:
        ari_leiden_na = None
    
    if data_.obs['louvain'][0] is not None:
        ari_louvain_na = metrics.adjusted_rand_score(data_.obs['louvain'], data_.obs['ground_truth'])
    else:
        ari_louvain_na = None
    
    ari_kmeans_na = metrics.adjusted_rand_score(data_.obs['kmeans'], data_.obs['ground_truth'])
    
    cols = ['mclust', 'leiden', 'louvain', 'kmeans']
    data_.obs[cols].to_csv(cluster_path + 'stLearn_na.tsv', sep='\t', index=True)
    
    # filter out NA nodes
    df = pd.read_csv(file_fold + '/' + sample + "_truth.txt", header=None, sep='\s+')
    data_.obs['domain_names'] = df.iloc[:, 1].values
    data_ = data_[~pd.isnull(data_.obs['domain_names'])]
    
    # calculate metric ARI
    ari_mclust = metrics.adjusted_rand_score(data_.obs['mclust'], data_.obs['ground_truth'])
    
    if data_.obs['leiden'][0] is not None:
        ari_leiden = metrics.adjusted_rand_score(data_.obs['leiden'], data_.obs['ground_truth'])
    else:
        ari_leiden = None
    
    if data_.obs['louvain'][0] is not None:
        ari_louvain = metrics.adjusted_rand_score(data_.obs['louvain'], data_.obs['ground_truth'])
    else:
        ari_louvain = None
    
    ari_kmeans = metrics.adjusted_rand_score(data_.obs['kmeans'], data_.obs['ground_truth'])
    
    data_.obs[cols].to_csv(cluster_path + 'stLearn.tsv', sep='\t', index=True)
    
    return ari_mclust_na, ari_leiden_na, ari_louvain_na, ari_kmeans_na, ari_mclust, ari_leiden, ari_louvain, ari_kmeans
    

############################################################################

if __name__ == '__main__':

   slice_name = None
   parser = argparse.ArgumentParser(description="Running stLearn")
   parser.add_argument('--slice', type=str, default='151507', help='name of the dataset')
   parser.add_argument('--n_clusters', type=int, default=5, help='number of clusters')
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

