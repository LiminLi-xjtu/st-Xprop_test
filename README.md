# st-Xprop: Comparison Methods

This repository provides standardized scripts and evaluation pipelines for reproducing the benchmark comparisons presented in the paper: **Cross-Propagative Graph Learning Reveals Spatial Tissue Domains in Multi-Modal Spatial Transcriptomics**

All comparison methods were evaluated against **st-Xprop** across multiple spatial transcriptomics datasets. Each method includes ready-to-run scripts using the default parameters reported in their official publications or implementations.

## Included Comparison Methods

| Method | Reference Implementation |
|--------|--------------------------|
| [DeepGFT](https://github.com/jxLiu-bio/DeepGFT) | jxLiu-bio/DeepGFT |
| [GraphST](https://github.com/JinmiaoChenLab/GraphST) | JinmiaoChenLab/GraphST |
| [SEDR](https://github.com/JinmiaoChenLab/SEDR) | JinmiaoChenLab/SEDR |
| [SpaGCN](https://github.com/jianhuupenn/SpaGCN) | jianhuupenn/SpaGCN |
| [SpaICL](https://github.com/wenwenmin/SpaICL) | wenwenmin/SpaICL |
| [SpaMask](https://github.com/wenwenmin/SpaMask) | wenwenmin/SpaMask |
| [spCLUE](https://github.com/EnchantedJoy/spCLUE) | EnchantedJoy/spCLUE |
| [STAGATE](https://github.com/zhanglabtools/STAGATE) | zhanglabtools/STAGATE |
| [stLearn](https://github.com/BiomedicalMachineLearning/stLearn) | BiomedicalMachineLearning/stLearn |

## Reproducing Benchmark Results

1. Navigate to the `compared_methods` folder:
   ```bash
   cd compared_methods
2. Run the script for the desired dataset:
   ```bash
   python <method_name>/run_<dataset>.py
   ```
   - Replace `method_name` with the desired method (e.g., `DeepGFT`, `GraphST`).
   - Replace `dataset.py` with the desired dataset (e.g., `DLPFC`, `BRCA`).
   Example:
   ```bash
   python DeepGFT/run_DLPFC.py
   ```

## Main Repository and Data Availability

The source code and datasets for the st-Xprop study are available at:  
- GitHub: [https://github.com/LiminLi-xjtu/st-Xprop](https://github.com/LiminLi-xjtu/st-Xprop)  
- Zenodo: [https://zenodo.org/records/18449464](https://zenodo.org/records/18449464)  

---  

For further details, refer to the documentation and supplementary materials in the repository.

