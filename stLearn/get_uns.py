import os
import numpy as np
import scanpy as sc
from PIL import Image
import stlearn as st
import matplotlib.pyplot as plt

def prepare_uns_spatial(adata, img_path, sample_name="sample1", lowres_scale=0.1):
    """
    构造 adata.uns['spatial']，同时处理灰度图 → RGB
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"{img_path} not found")
    
    # 载入图像并转 RGB
    hires_img = Image.open(img_path).convert("RGB")
    hires_array = np.array(hires_img)
    
    # lowres 图像
    lowres_img = hires_img.resize(
        (hires_img.width // 10, hires_img.height // 10)
    )
    lowres_array = np.array(lowres_img)

    adata.uns["spatial"] = {
        sample_name: {
            "images": {
                "hires": hires_array,
                "lowres": lowres_array,
            },
            "scalefactors": {
                "spot_diameter_fullres": 100,  # 需根据实际坐标单位调整
                "tissue_hires_scalef": 1.0,
                "tissue_lowres_scalef": lowres_scale,
            },
        }
    }
    return adata


import numpy as np
import pandas as pd
import stlearn as st
from scipy.sparse import csr_matrix
def SME_normalize_for_stereo(adata, use_data="raw", weights_key="weights_matrix_all"):
    # 1) 准备 count_embed 与检查类型（仿照 SME_normalize）
    if use_data == "raw":
        if isinstance(adata.X, csr_matrix):
            count_embed = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            count_embed = adata.X
        elif isinstance(adata.X, pd.DataFrame):
            count_embed = adata.X.values
        else:
            raise ValueError(f"{type(adata.X)} is not a valid type.")
    else:
        if use_data not in adata.obsm:
            raise KeyError(f"{use_data} not found in adata.obsm")
        count_embed = adata.obsm[use_data]

    # 2) 计算权重矩阵（我们自定义的函数）
    # 这会把 adata.uns["weights_matrix_all"] 填好
    adata = calculate_weight_matrix_stereo(adata, use_morphology=False, use_expression=True, distance_scale=1.0)

    # 3) 调用 impute_neighbour（和 SME_normalize 内部一致）
    # 注意：impute_neighbour 的位置可能在 st.spatial.SME 模块中
    impute_neighbour(adata, count_embed=count_embed, weights=weights_key)

    # 4) 取回 imputed_data 并合并（与 SME_normalize 行为一致）
    imputed_data = adata.obsm["imputed_data"].astype(float)
    imputed_data[imputed_data == 0] = np.nan
    adjusted_count_matrix = np.nanmean(np.array([count_embed, imputed_data]), axis=0)

    key_added = use_data + "_SME_normalized"
    adata.obsm[key_added] = adjusted_count_matrix

    print("Stereo SME finished. Result in adata.obsm['" + key_added + "']")
    return adata



import numpy as np
from sklearn.metrics import pairwise_distances
from anndata import AnnData


def calculate_weight_matrix_stereo(
    adata: AnnData,
    use_morphology: bool = False,
    use_expression: bool = True,
    distance_scale: float = 1.0,
):
    """
    兼容 Stereo-seq 的 stLearn SME 加权矩阵计算函数。
    仅基于空间距离（可选结合形态学或表达相似度）。

    参数:
    ----------
    adata : AnnData
        包含空间坐标的 AnnData 对象，要求 adata.obsm["spatial"] 存在。
    use_morphology : bool, optional
        是否使用 adata.obsm["X_morphology"] 参与加权。
        默认为 False，因为 Stereo-seq 通常没有 RGB 图像特征。
    use_expression : bool, optional
        是否使用 adata.obsm["X_pca"] 参与基因表达相似度加权。
    distance_scale : float, optional
        控制空间衰减范围（越大 -> 邻域越宽）。默认 1.0。
    """

    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required for Stereo-seq data.")

    coords = adata.obsm["spatial"]

    # --- 1. 空间距离矩阵 ---
    pd = pairwise_distances(coords, metric="euclidean")
    # 归一化为 [0,1] 的相似度形式
    pd_norm = np.exp(-pd / (np.std(pd) * distance_scale))
    adata.uns["physical_distance"] = pd_norm

    # --- 2. 可选: 形态学特征相似度 ---
    if use_morphology and "X_morphology" in adata.obsm:
        md = 1 - pairwise_distances(adata.obsm["X_morphology"], metric="cosine")
        md[md < 0] = 0
        adata.uns["morphological_distance"] = md
    else:
        md = np.ones_like(pd_norm)
        adata.uns["morphological_distance"] = None

    # --- 3. 可选: 基因表达特征相似度 ---
    if use_expression and "X_pca" in adata.obsm:
        gd = 1 - pairwise_distances(adata.obsm["X_pca"], metric="correlation")
        gd[gd < 0] = 0
        adata.uns["gene_expression_correlation"] = gd
    else:
        gd = np.ones_like(pd_norm)
        adata.uns["gene_expression_correlation"] = None

    # --- 4. 组合加权矩阵 ---
    weights_matrix = pd_norm * md * gd
    adata.uns["weights_matrix_all"] = weights_matrix

    print(
        f"✅ Stereo-seq weight matrix computed "
        f"(spatial only: {not use_morphology and not use_expression})"
    )
    return adata



from sklearn.metrics import pairwise_distances
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from stlearn._compat import Literal
from tqdm import tqdm

_WEIGHTING_MATRIX = Literal[
    "weights_matrix_all",
    "weights_matrix_pd_gd",
    "weights_matrix_pd_md",
    "weights_matrix_gd_md",
    "gene_expression_correlation",
    "physical_distance",
    "morphological_distance",
]



def impute_neighbour(
    adata: AnnData,
    count_embed: Union[np.ndarray, None] = None,
    weights: _WEIGHTING_MATRIX = "weights_matrix_all",
    copy: bool = False,
) -> Optional[AnnData]:
    coor = adata.obs[["imagecol", "imagerow"]]

    weights_matrix = adata.uns[weights]

    lag_coor = []

    weights_list = []

    with tqdm(
        total=len(adata),
        desc="Adjusting data",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for i in range(len(coor)):

            main_weights = weights_matrix[i]

            if weights == "physical_distance":
                current_neighbour = main_weights.argsort()[-6:]
            else:
                current_neighbour = main_weights.argsort()[-3:]

            surrounding_count = count_embed[current_neighbour]
            surrounding_weights = main_weights[current_neighbour]
            if surrounding_weights.sum() > 0:
                surrounding_weights_scaled = (
                    surrounding_weights / surrounding_weights.sum()
                )
                weights_list.append(surrounding_weights_scaled)

                surrounding_count_adjusted = np.multiply(
                    surrounding_weights_scaled.reshape(-1, 1), surrounding_count
                )
                surrounding_count_final = np.sum(surrounding_count_adjusted, axis=0)

            else:
                surrounding_count_final = np.zeros(count_embed.shape[1])
                weights_list.append(np.zeros(len(current_neighbour)))
            lag_coor.append(surrounding_count_final)
            pbar.update(1)

    imputed_data = np.array(lag_coor)
    key_added = "imputed_data"
    adata.obsm[key_added] = imputed_data

    adata.obsm["top_weights"] = np.array(weights_list)

    return adata if copy else None

