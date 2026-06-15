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


def main(args, num_clusters):


    sample = args.slice
    n_clusters = args.n_clusters

    BASE_PATH = Path('../../../datasets/'+sample)
    file_fold = '../../../datasets/' + sample

    # dir_input = f'../data/DLPFC/{sample}/'
    # dir_output = f'../output/{sample}/stLearn/'
    #
    # if not os.path.exists(dir_output):
    #     os.makedirs(dir_output)
    

    ##################################################################################
    result_path = '../output/' + sample + '/'
    cluster_path = '../cluster/' + sample + '/stLearn/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    ##################################################################################

    
    # n_clusters=7
    # clu=str(n_clusters)
    TILE_PATH = Path("./tmp/{}_tiles".format(sample))
    TILE_PATH.mkdir(parents=True, exist_ok=True)
    
    # OUTPUT_PATH = Path(f"../output/DLPFC/{sample}/stLearn")
    # OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    data = st.Read10X(BASE_PATH, count_file=str(sample)+'_filtered_feature_bc_matrix.h5')
    data  = st.convert_scanpy(data)
    
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
    # st.spatial.SME.SME_normalize(data, use_data="raw", weights="weights_matrix_pd_md") #, weights="physical_distance"
    data_ = data.copy()
    data_.X = data_.obsm['raw_SME_normalized']
    st.pp.scale(data_)
    st.em.run_pca(data_,n_comps=50)
    
    emb = data_.obsm['X_pca']
    np.save(result_path + 'stLearn.npy', emb)

    ############################################################################################################
    radius = 50
    used_obsm = 'X_pca'

    columns = ['n_clusters', 'kmeans', 'mclust', 'louvain', 'leiden']
    rows = [str(i) for i in num_clusters]
    results_df = pd.DataFrame(columns=columns)

    for n in range(len(num_clusters)):
        n_clusters = num_clusters[n]

        clustering(data_, n_clusters, radius=radius, used_obsm=used_obsm, method='mclust')

        try:
            clustering(data_, n_clusters, radius=radius, used_obsm=used_obsm, method='leiden', start=0.1, end=3.0,
                       increment=0.01)
        except Exception as e:
            data_.obs['leiden'] = None

        try:
            clustering(data_, n_clusters, radius=radius, used_obsm=used_obsm, method='louvain', start=0.1, end=3.0,
                       increment=0.01)
        except Exception as e:
            data_.obs['louvain'] = None

        clustering(data_, n_clusters, radius=radius, used_obsm=used_obsm, method='kmeans')

        silh_mclust = metrics.silhouette_score(data_.obsm[used_obsm], data_.obs['mclust'])

        if data_.obs['leiden'][0] is not None:
            silh_leiden = metrics.silhouette_score(data_.obsm[used_obsm], data_.obs['leiden'])
        else:
            silh_leiden = None

        if data_.obs['louvain'][0] is not None:
            silh_louvain = metrics.silhouette_score(data_.obsm[used_obsm], data_.obs['louvain'])
        else:
            silh_louvain = None

        silh_kmeans = metrics.silhouette_score(data_.obsm[used_obsm], data_.obs['kmeans'])

        cols = ['mclust', 'leiden', 'louvain', 'kmeans']
        data_.obs[cols].to_csv(cluster_path + 'clu_' + str(n_clusters) + '.tsv', sep='\t', index=True)

        silhs = {'n_clusters': str(n_clusters), 'mclust': silh_mclust, 'leiden': silh_leiden, 'louvain': silh_louvain,
                 'kmeans': silh_kmeans}
        results_df = pd.concat([results_df, pd.DataFrame([silhs])], ignore_index=True)

    return results_df


############################################################################

if __name__ == '__main__':

   slice_name = None
   parser = argparse.ArgumentParser(description="Running stLearn")
   parser.add_argument('--slice', type=str, default='MBP', help='name of the dataset')
   parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
   args = parser.parse_args()
   slice_name = args.slice

   num_clusters = np.arange(stop=26, start=6, step=1)

   # num_clusters = np.arange(stop=25, start=6, step=1)

   results_df = main(args, num_clusters)

   eva_path = '../evaluation/' + slice_name + '/'

   results_df.to_csv(eva_path + "/stLearn.csv", index=False)