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

    dir_input = '../../datasets/CHD/' + str(sample_name)   # please replace 'file_fold' with the download path
    dir_output = f'./output/CHD/{sample_name}/'
    cluster_path = './cluster/CHD/' + sample_name + '/'

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    # read data
    # adata = sc.read_visium(dir_input, count_file=str(sample_name) + '_filtered_feature_bc_matrix.h5', load_images=True)

    adata = sc.read_h5ad(f'{dir_input}/{sample_name}_reset.h5ad')
    adata.obs['ground_truth'] = adata.obs['region']
    n_clusters = len(np.unique(adata.obs['ground_truth']))

            
    adata.var_names_make_unique()


    if args.slice=='D14_estimated' or args.slice=='D14_measured':
        spatial=pd.read_csv("../../datasets/CHD/D14/spatial/tissue_positions.csv",sep=",",header=0,na_filter=False)
    else:
        spatial=pd.read_csv(f"{dir_input}/spatial/tissue_positions.csv",sep=",",header=0,na_filter=False) #,index_col=0

    spatial.columns = ['barcode', 'in_tissue', 'col', 'row', 'pxl_y', 'pxl_x']
    spatial['in_tissue'] = spatial['in_tissue'].astype(int)
    spatial = spatial[spatial['in_tissue'] == 1].reset_index(drop=True)
    spatial["barcode"] = spatial["barcode"].str.replace("-1", "", regex=False)
    spatial_df = spatial.set_index("barcode")
    spatial_df = spatial_df.loc[adata.obs_names]
    # spatial_df = spatial_df.reset_index()

    adata.obs["x1"]=spatial_df['in_tissue'].astype('int')
    adata.obs["x2"]=spatial_df['row'].astype('int')
    adata.obs["x3"]=spatial_df['col'].astype('int')
    adata.obs["x4"]=spatial_df['pxl_x'].astype('int')
    adata.obs["x5"]=spatial_df['pxl_y'].astype('int')

    adata.var_names=[i.upper() for i in list(adata.var_names)]
    adata.var["genename"]=adata.var.index.astype("str")
    # adata.write_h5ad(f"{dir_output}/sample_data.h5ad")


    #Read in hitology image
    if args.slice=='D14_estimated' or args.slice=='D14_measured':
        img=cv2.imread('../../datasets/CHD/D14/chicken_heart_spatial_RNAseq_D14_image.tif')
    else:
        img=cv2.imread(f'{dir_input}/chicken_heart_spatial_RNAseq_{sample_name}_image.tif')


    #Set coordinates
    adata.obs["x_array"]=adata.obs["x2"]
    adata.obs["y_array"]=adata.obs["x3"]
    adata.obs["x_pixel"]=adata.obs["x4"]
    adata.obs["y_pixel"]=adata.obs["x5"]
    x_array=adata.obs["x_array"].tolist()
    y_array=adata.obs["y_array"].tolist()
    x_pixel=adata.obs["x_pixel"].tolist()
    y_pixel=adata.obs["y_pixel"].tolist()

    #Test coordinates on the image
    img_new=img.copy()
    for i in range(len(x_pixel)):
        x=x_pixel[i]
        y=y_pixel[i]
        img_new[int(x-20):int(x+20), int(y-20):int(y+20),:]=0

    # cv2.imwrite(f'{dir_output}/sample_map.jpg', img_new)

    #Calculate adjacent matrix
    b=49
    a=1
    adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=a, histology=True)
    # np.savetxt(f'{dir_output}/adj.csv', adj, delimiter=',')



    ##### Spatial domain detection using SpaGCN
    spg.prefilter_genes(adata, min_cells=3) # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)


    ### 4.2 Set hyper-parameters

    p = 0.5
    # Find the l value given p
    l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    n_clusters=n_clusters
    r_seed=t_seed=n_seed=100
    # res=spg.search_res(adata, adj, l, n_clusters, start=0.8, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed,
    #     t_seed=t_seed, n_seed=n_seed) #D10, D14
    res=spg.search_res(adata, adj, l, n_clusters, start=0.62, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed,
        t_seed=t_seed, n_seed=n_seed) #D4, D7
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
    adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
    refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
    # np.savetxt(dir_output + clu+'refined_pred.csv', y_pred, delimiter=",")
    #Save results
    # adata.write_h5ad(f"{dir_output}/results.h5ad")

    # adata.obs.to_csv(f'{dir_output}/metadata.tsv', sep='\t')

    cols = ['pred', 'refined_pred']
    adata.obs[cols].to_csv(cluster_path + 'SpaGCN.tsv', sep='\t', index=True)
    
    print(len(np.unique(y_pred)))
    print(len(np.unique(refined_pred)))

    # df = pd.read_csv(dir_input + '/' + sample_name + "_truth.txt", header=None, sep='\s+')
    # adata.obs['ground_truth'] = df.iloc[:, 1].values

    ari_louvain = metrics.adjusted_rand_score(adata.obs['pred'], adata.obs['ground_truth'])
    ref_ari_louvain = metrics.adjusted_rand_score(adata.obs['refined_pred'], adata.obs['ground_truth'])


    return ari_louvain, ref_ari_louvain



if __name__ == '__main__':

   slice_name = None
   parser = argparse.ArgumentParser(description="Running SpaGCN")
   parser.add_argument('--slice', type=str, default='D4', help='name of the dataset') #'D14_estimated' 'D14_measured'
#    parser.add_argument('--n_clusters', type=int, default=20, help='number of clusters')
   args = parser.parse_args()
   slice_name = args.slice

   ari_louvain, ref_ari_louvain = main(args)

   import pandas as pd

   results = {
      'slice': [slice_name],
      'ari_louvain': [ari_louvain],
      'ref_ari_louvain': [ref_ari_louvain]
   }
   df = pd.DataFrame(results)
   # print(df.to_csv(index=False))

   eva_path = './evaluation/CHD/' + slice_name + '/'

   if not os.path.exists(eva_path):
       os.makedirs(eva_path)

   df.to_csv(eva_path + "/SpaGCN.csv", index=False)

   # import sys
   # df.to_csv(sys.stdout, index=False)



