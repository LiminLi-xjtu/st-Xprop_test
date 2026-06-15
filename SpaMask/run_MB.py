from prepare import *
import argparse

# os.environ['R_HOME'] = '/users/PCON0022/jxliu/anaconda3/envs/SpaMask/lib/R'

def build_args_():
    import argparse
    parser = argparse.ArgumentParser(description="stMask")
    parser.add_argument("--model_name", type=str, default="SpaMask")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--tissue_name", type=str, default="151507")

    parser.add_argument("--top_genes", type=int, default=2000)
    parser.add_argument("--genes_model", type=str, default="pca")
    parser.add_argument("--rad_cutoff", type=int, default=200)
    parser.add_argument("--k_cutoff", type=int, default=12)
    parser.add_argument("--graph_model", type=str, default="KNN")

    parser.add_argument('--nps', type=int, default=30)
    parser.add_argument('--gradient_clipping', type=float, default=5.)
    parser.add_argument("--need_refine", action='store_true', default=False)

    # 各模型的训练设置
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--max_epoch", type=int, default=500, help="number of training epochs")

    # ST params
    parser.add_argument("--edge_drop_rate", type=float, default=0.4)
    parser.add_argument("--feat_mask_rate", type=float, default=0.3)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=256)

    parser.add_argument('--bn', action='store_true', default=True)
    parser.add_argument("--att_dropout_rate", type=float, default=.2)
    parser.add_argument("--fc_dropout_rate", type=float, default=.5)
    parser.add_argument("--use_token", action='store_true', default=True)
    parser.add_argument("--rep_loss", type=str, default="cse")
    parser.add_argument("--rel_loss", type=str, default="ce")
    parser.add_argument("--alpha", type=float, default=2.0)

    parser.add_argument("--lam", type=float, default=2)
    # args = parser.parse_args(args=[])
    return parser



def load_adata(section_ids, k_cutoff, rad_cutoff, model, n_top_genes):
    Batch_list = []
    for section_id in section_ids:
        print(section_id)

        # Read data
        path = '../../datasets/MB/' + str(section_id)  # please replace 'file_fold' with the download path
        adata = sc.read_visium(path, count_file=str(section_id) + '_filtered_feature_bc_matrix.h5')
             
        adata.var_names_make_unique()
        from scipy.sparse import csr_matrix
        adata.X = csr_matrix(adata.X)

        
        # make spot name unique
        # adata.obs_names = [x + '_' + section_id for x in adata.obs_names]

        # stm.Cal_Spatial_Net(adata, rad_cutoff=150)
        adata.var_names_make_unique()
        adata.layers['count'] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top_genes)
        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)
        adata = adata[:, adata.var['highly_variable']]
        Batch_list.append(adata)

    # %%
    Batch_list = align_spots(Batch_list, method='icp', plot=False)
    # %%
    adata_st = preprocess(Batch_list, section_ids=section_ids, k_cutoff=k_cutoff, rad_cutoff=rad_cutoff, model=model,
                          slice_dist_micron=None) #[10, 10, 10]
    adata_X = PCA(n_components=200, random_state=42).fit_transform(adata_st.X)
    adata_st.obsm['feat'] = adata_X

    return adata_st



def main(args, num_clusters):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    

    proj_name = args.slice
    slices_list = [proj_name]   
    result_path = './output/' + proj_name + '/SpaMask/'
    cluster_path = './cluster/' + proj_name + '/SpaMask/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    # Read data


    ############################################################################################################

    columns = ['n_clusters', 'kmeans']
    rows = [str(i) for i in num_clusters]
    results_df = pd.DataFrame(columns=columns)
    
    for n in range(len(num_clusters)):
        n_clusters = num_clusters[n]
        
        adata = load_adata(slices_list, k_cutoff=args.k_cutoff, rad_cutoff=args.rad_cutoff, model=args.model,
                n_top_genes=args.top_genes)
        adata = train_one(args, adata, n_clusters, device)
        
        np.save(result_path + 'clu_' + str(n_clusters) +'.npy', adata.obsm['eval_pred'])


        silh_kmeans =  metrics.silhouette_score(adata.obsm['eval_pred'], adata.obs['kmeans'])

        cols = ['kmeans']
        adata.obs[cols].to_csv(cluster_path + 'clu_' + str(n_clusters) + '.tsv', sep='\t', index=True)

        silhs = {'n_clusters': str(n_clusters), 'kmeans': silh_kmeans}
        results_df = pd.concat([results_df, pd.DataFrame([silhs])], ignore_index=True)

    return results_df



if __name__ == '__main__':
    parser = build_args_()
    # args.hidden_dim, args.latent_dim = 512, 256
    # args.max_epoch = 1000
    # args.lam = 2
    # args.feat_mask_rate = 0.5
    # args.edge_drop_rate = 0.2
    # args.top_genes = 5000
    # args.rad_cutoff = 200
    # args.k_cutoff = 21
    # args.model = 'KNN'
    # args.name='DLPFC'
    # args.slice='151673'
    
    
    # parser = argparse.ArgumentParser(description="Running SpaMask")
    parser.add_argument('--slice', type=str, default='MBP', help='name of the dataset')
    # parser.add_argument('--n_clusters', type=int, default=7, help='number of clusters')
    args = parser.parse_args()
    args.hidden_dim, args.latent_dim = 512, 256
    args.max_epoch = 1000
    args.lam = 2
    args.feat_mask_rate = 0.5
    args.edge_drop_rate = 0.2
    args.top_genes = 5000
    args.rad_cutoff = 200
    args.k_cutoff = 21
    args.model = 'KNN'

    num_clusters = [10, 12, 16, 18, 20, 22, 25] # np.arange(stop=26, start=6, step=1)
    # ari_kmeans = main(args, num_clusters) #ari_mclust_na, ari_leiden_na, ari_louvain_na, ari_kmeans_na, 

    results_df = main(args, num_clusters)

    eva_path = './evaluation/' + args.slice + '/'

    results_df.to_csv(eva_path + "/SpaMask.csv", index=False)


