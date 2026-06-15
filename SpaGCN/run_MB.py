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


def main(args, num_clusters, res_starts, res_list):
    sample_name = args.slice
    # n_clusters = args.n_clusters

    dir_input = '../../datasets/' + str(sample_name) + '/'  # please replace 'file_fold' with the download path

    ##################################################################################
    dir_output = './output/' + sample_name + '/SpaGCN/'
    cluster_path = './cluster/' + sample_name + '/SpaGCN/'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    ##################################################################################


    # read data
    adata = sc.read_visium(dir_input, count_file=str(sample_name) + '_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()

    spatial=pd.read_csv(f"{dir_input}/spatial/tissue_positions_list.csv",sep=",",header=None,na_filter=False,index_col=0)
    adata.obs["x1"]=spatial[1]
    adata.obs["x2"]=spatial[2]
    adata.obs["x3"]=spatial[3]
    adata.obs["x4"]=spatial[4]
    adata.obs["x5"]=spatial[5]

    adata.var_names=[i.upper() for i in list(adata.var_names)]
    adata.var["genename"]=adata.var.index.astype("str")
    # adata.write_h5ad(f"{dir_output}/sample_data.h5ad")


    #Read in hitology image
    img=cv2.imread(dir_input + sample_name+"_full_image.tif")

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

    columns = ['n_clusters', 'louvain', 'ref_louvain']
    rows = [str(i) for i in num_clusters]
    results_df = pd.DataFrame(columns=columns)

    res_list = []
    num_pred = []
    num_pred_ref = []
    for n in range(len(num_clusters)):
        n_clusters = num_clusters[n]
        r_seed=t_seed=n_seed=100

        res_start = res_starts[n]
        res=spg.search_res(adata, adj, l, n_clusters, start=res_start, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed,
            t_seed=t_seed, n_seed=n_seed)
        res_list.append(res)

        # res = res_list[n]
        

        ### 4.3 Run SpaGCN
        clf=spg.SpaGCN()
        clf.set_l(l)
        #Set seed
        random.seed(r_seed)
        torch.manual_seed(t_seed)
        np.random.seed(n_seed)
        #Run
        clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
        em = clf.embed

        clu=str(n_clusters)
        np.save(dir_output + 'clu_' + str(n_clusters) + '.npy', em)
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
        
        num_pred.append(len(np.unique(y_pred)))
        num_pred_ref.append(len(np.unique(refined_pred)))

        cols = ['pred', 'refined_pred']
        adata.obs[cols].to_csv(cluster_path + 'clu_' + str(n_clusters) + '.tsv', sep='\t', index=True)


        # ari_louvain_na = metrics.adjusted_rand_score(adata.obs['pred'], adata.obs['ground_truth'])
        # ref_ari_louvain_na = metrics.adjusted_rand_score(adata.obs['refined_pred'], adata.obs['ground_truth'])

        silh_louvain = metrics.silhouette_score(em, adata.obs['pred'])
        silh_ref_louvain = metrics.silhouette_score(em, adata.obs['refined_pred'])

        silhs = {'n_clusters': str(n_clusters), 'louvain': silh_louvain, 'ref_louvain': silh_ref_louvain}
        results_df = pd.concat([results_df, pd.DataFrame([silhs])], ignore_index=True)


    return results_df



if __name__ == '__main__':

    slice_name = None
    parser = argparse.ArgumentParser(description="Running SpaGCN")
    parser.add_argument('--slice', type=str, default='MBC', help='name of the dataset')
    parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
    args = parser.parse_args()
    slice_name = args.slice

    num_clusters = np.arange(stop=26, start=7, step=1)
    
    # MBP, 15 not available
    # res_starts = [0.2, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.6, 0.7, 0.7, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0, 1.3, 1.4, 1.5] # 
    # res_list = [0.15, 0.175, 0.2, 0.225, 0.4, 0.475, 0.5, 0.6, 0.65, 0.6921875, 0.7, 0.8, 0.9, 0.925, 0.9375, 0.95, 1.0, 1.3, 1.4, 1.41015625]
    # res_list = [0.2, 0.25, 0.3, 0.5, 0.55, #7-11
    #             0.7, 0.75, 0.8, 0.9,  #12-15
    #             1.0, 1.0125, 1.2, 1.25, 1.2521875, #16-20
    #             1.26, 1.3, 1.4, 1.5, 1.6] #21-25
    # res_starts=[0.2, 0.25, 0.3, 0.5, 0.5, #7-11
    #             0.7, 0.75, 0.8, 0.9,  #12-15
    #             1.0, 1.0125, 1.2, 1.25, 1.2521875, #16-20
    #             1.26, 1.3, 1.4, 1.5, 1.6]


    
    
    # MBC
    # res_starts = [0.2, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.6, 0.7, 0.7, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0, 1.3, 1.4, 1.5]
    # res_list = [0.2, 0.25, 0.275, 0.3, 0.4, 0.4375, 0.45, 0.5, 0.6, 0.775, 0.784375,
    #              0.8, 0.9, 0.95, 0.9625, 1.0, 1.075, 1.1, 1.2, 1.3] # 17-25
    res_starts = [0.2, 0.3, 0.4, 0.4, 0.45, 
                  0.5, 0.5, 0.525, 0.55, 0.7, 
                  0.75, 0.75, 0.95, 1.1, 1.15, 
                  1.15, 1.45, 1.75, 1.8]
    res_list = [0.3, 0.35, 0.4, 0.45, 0.478125, #7-11
                0.5, 0.5125, 0.525, 0.55, 0.7328125, #12-16
                0.75, 0.95, 1.0875, 1.1, 1.15, #17-21
                1.45, 1.75, 1.775, 1.8] #22-25
    

    results_df = main(args, num_clusters, res_starts, res_list)

    eva_path = './evaluation/' + slice_name + '/'

    results_df.to_csv(eva_path + "/SpaGCN.csv", index=False)



