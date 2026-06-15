import os,csv,re,sys
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
import random, torch
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import argparse
from sklearn import metrics


def main(args):
    sample_name = args.slice
    # n_clusters = args.n_clusters

    dir_input = '../../datasets/HER2ST/' + str(sample_name)  # please replace 'file_fold' with the download path
    dir_output = f'./output/HER2ST/{sample_name}/'
    cluster_path = './cluster/HER2ST/' + sample_name + '/'

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    # read data
    # adata = sc.read_visium(dir_input, count_file=str(sample_name) + '_filtered_feature_bc_matrix.h5', load_images=True)
    # adata.var_names_make_unique()

    adata = sc.read_h5ad(dir_input+'.h5ad')
    from scipy.sparse import csr_matrix
    adata.X = csr_matrix(adata.X)

    adata.obs['ground_truth'] = adata.obs['annotation']
    n_clusters = len(np.unique(adata.obs['ground_truth']))-1

    adata.var_names=adata.var.index.astype("str")
    adata.var_names_make_unique()
    spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    adata.X = adata.X.astype(np.float64)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    adata.obs['x_pixel']=adata.obsm['spatial'][:,0]
    adata.obs['y_pixel']=adata.obsm['spatial'][:,1]

    x_pixel=adata.obs["x_pixel"].tolist()
    y_pixel=adata.obs["y_pixel"].tolist()

    #Calculate adjacent matrix
    s=1
    b=49
    
    #If histlogy image is not available, SpaGCN can calculate the adjacent matrix using the fnction below
    adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
    # np.savetxt(f'./adj.csv', adj, delimiter=',')
    #spatial domain detection using SpaGCN

    #expression data preprocessing
    # adj=np.loadtxt(f'./adj.csv', delimiter=',')
    
    #set hyper-parameters
    p=0.5 
    #Find the l value given p
    l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    # n_clusters=n_clusters
    r_seed=t_seed=n_seed=100
    res=spg.search_res(adata, adj, l, n_clusters, start=0.8, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed,
        t_seed=t_seed, n_seed=n_seed)

    print(res)



    ### 4.3 Run SpaGCN
    clf=spg.SpaGCN()
    clf.set_l(l)
    #Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    #Run
    clf.train(adata, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=200)
    em = clf.embed

    clu=str(n_clusters)
    np.save(dir_output + 'SpaGCN.npy', em)
    # np.savetxt(fname=dir_output + 'SpaGCN_clu_' + clu + '.csv', X=em, fmt="%s",delimiter=",")


    y_pred, prob=clf.predict()
    adata.obs["pred"]= y_pred
    adata.obs["pred"]=adata.obs["pred"].astype('category')

    #pca_ari = adjusted_rand_score(pred, gt)

    #Do cluster refinement(optional)
    x_array=adata.obs["array_row"].tolist()
    y_array=adata.obs["array_col"].tolist()
    adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
    refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
    # np.savetxt(dir_output + clu+'refined_pred.csv', y_pred, delimiter=",")
    #Save results
    # adata.write_h5ad(f"{dir_output}/results.h5ad")

    # adata.obs.to_csv(f'{dir_output}/metadata.tsv', sep='\t')

    cols = ['pred','refined_pred']
    adata.obs[cols].to_csv(cluster_path + 'SpaGCN.tsv', sep='\t', index=True)


    adata2 = adata[adata.obs['ground_truth'] != 'undetermined'].copy()
    ari_louvain = metrics.adjusted_rand_score(adata2.obs['pred'], adata2.obs['ground_truth'])
    ref_ari_louvain = metrics.adjusted_rand_score(adata2.obs['refined_pred'], adata2.obs['ground_truth'])


    return ari_louvain, ref_ari_louvain




if __name__ == '__main__':

   slice_name = None
   parser = argparse.ArgumentParser(description="Running SpaGCN")
   parser.add_argument('--slice', type=str, default='C1', help='name of the dataset')
   parser.add_argument('--n_clusters', type=int, default=4, help='number of clusters')
   args = parser.parse_args()
   slice_name = args.slice

   ari_louvain,ref_ari_louvain = main(args)

   import pandas as pd

   results = {
      'slice': [slice_name],
      'ari_louvain': [ari_louvain],
      'ref_ari_louvain': [ref_ari_louvain]
   }
   df = pd.DataFrame(results)
   # print(df.to_csv(index=False))

#    eva_path = './evaluation/HER2ST/' + slice_name + '/'

#    if not os.path.exists(eva_path):
#        os.makedirs(eva_path)

#    df.to_csv(eva_path + "/SpaGCN.csv", index=False)

   import sys
   df.to_csv(sys.stdout, index=False)



