
# import sys
# import os
# from pathlib import Path
# import torch
# import torch.nn as nn

# 添加方法路径
# BASE_DIR = Path(__file__).parent.parent  # comp_methods
# sys.path.append(str(BASE_DIR / 'DeepGFT-main'))
# sys.path.append(str(BASE_DIR / 'GraphST'))
# sys.path.append(str(BASE_DIR / 'SEDR-main'))  
# sys.path.append(str(BASE_DIR / 'SpaBatch-main'))
# sys.path.append(str(BASE_DIR / 'SpaCross-main'))
# sys.path.append(str(BASE_DIR / 'SpaGCN'))  
# sys.path.append(str(BASE_DIR / 'SpaICL-main'))
# sys.path.append(str(BASE_DIR / 'SpaMask-main'))
# sys.path.append(str(BASE_DIR / 'SpatialCVGAE-main'))  
# sys.path.append(str(BASE_DIR / 'spCLUE-main'))
# sys.path.append(str(BASE_DIR / 'STAGATE'))
# sys.path.append(str(BASE_DIR / 'stLearn-main'))  
# sys.path.append(str(BASE_DIR / 'st-Xprop'))  

dataset_name, dataset_slice = 'DLPFC', '151507'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 导入各方法模型
from DeepGFT_gt import run_DeepGFT
run_DeepGFT(dataset_name, dataset_slice)

import DeepGFT_wgt

# from model_code import SpaMaskModel, SpaCrossModel, STXPropModel  # 假设类名在 model_code.py

# 封装成统一函数，返回字典
def load_models(device='cuda'):
    models = {
        "st-Xprop": STXPropModel().to(device),
        "SpaMask": SpaMaskModel().to(device),
        "SpaCross": SpaCrossModel().to(device)
    }
    return models