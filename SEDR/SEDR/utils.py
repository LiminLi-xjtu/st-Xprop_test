import math
import itertools
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import torch
from torch.backends import cudnn
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import random



def refine_label(adata, radius=30, key='label', suffix=None):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    #calculate distance
    position = adata.obsm['spatial']
    distance = cdist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    suffix_add = "" if suffix is None else "_" + suffix
    new_type = [str(i) for i in list(new_type)]
    adata.obs[f'{key}_refined' + suffix_add] = np.array(new_type)
    return np.array(new_type)



def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res


def search_res_list(adata, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    
    from collections import defaultdict
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    res_of_clu = defaultdict(list) 
    
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
            
        res_of_clu[count_unique].append(res)   # 每一簇数都记录全部能达到的 res
        print(f'resolution={res:.3f}, cluster number={count_unique}')

    return res_of_clu  


def clustering(adata, n_clusters=7, radius=50, used_obsm='X_pca', method='mclust', start=0.1, end=3.0, increment=0.01, res_test=None, 
               refinement=False):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1.
    end : float
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """

    # pca = PCA(n_components=20, random_state=42)
    # embedding = pca.fit_transform(adata.obsm['emb'].copy())
    # adata.obsm['emb_pca'] = embedding

    if method == 'mclust':
        try:
            adata = mclust_R(adata, used_obsm=used_obsm, num_cluster=n_clusters)
        except Exception as e:
            print(f'Mclust on {used_obsm} failed ({e}); fallback to PCA(5) → mclust')
            from sklearn.decomposition import PCA
            embedding = PCA(n_components=10, random_state=42)\
                        .fit_transform(adata.obsm[used_obsm])
            adata.obsm['emb_pca'] = embedding
            adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust'].astype('category')
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(adata.obsm[used_obsm])
        adata.obs['kmeans'] = kmeans.labels_.astype(str)
    elif method == 'leiden':
        if res_test is None:
            res = search_res(adata, n_clusters, use_rep=used_obsm, method=method, start=start, end=end, increment=increment)
        else:
            res = res_test
            sc.pp.neighbors(adata, n_neighbors=50, use_rep=used_obsm)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        if res_test is None:
            res = search_res(adata, n_clusters, use_rep=used_obsm, method=method, start=start, end=end, increment=increment)
        else:
            res = res_test
            sc.pp.neighbors(adata, n_neighbors=50, use_rep=used_obsm)        
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain']

    if refinement:
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='used_obsm', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    robjects.r.library("mclust")

    # rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    # res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    # mclust_res = np.array(res[-2])

    with localconverter(robjects.default_converter + numpy2ri.converter):
        r_data = robjects.conversion.py2rpy(adata.obsm[used_obsm])
        res = rmclust(r_data, num_cluster, modelNames)
    # mclust_res = np.array(res.rx2('classification'))
    classification_idx = list(res.names()).index('classification')
    mclust_res = np.array(res[classification_idx])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


