# MET-MAP (Metabolic Topography Mapper)
<!-- **Reproducibility repository for Multi-GASTON applied on spatial metabolomics data.** -->

This repository provides the source code and Jupyter notebook tutorials for applying **MET-MAP**—a deep learning model that ecovers tissue architecture and spatial patterns of feature variation—to spatial metabolomics data from murine liver and small intestine. 
<p align="center">
<img src="plots/liver_demo.png" height=400/>
<img src="plots/intestine_demo.png" height=400/>
</p>

---

> [!IMPORTANT]
> **Version Note:** The source code contained in this repository is a **preliminary version** of the method used specifically for the analyses in our 2025 *Nature* publication. 
> 
> For the **current and actively maintained** version of the framework that enables simoutaneous learning from __multiple samples__ with __non-linear__ feature functions, please refer to [Multi-GASTON repository](https://github.com/raphael-group/Multi-GASTON).

---

## Overview

MET-MAP is an unsupervised deep learning model that learns multiple spatial gradients simoutaineously from spatial metabolomics data. It is an extension of GASTON (https://pmc.ncbi.nlm.nih.gov/articles/PMC10592770/), which was designed for SRT data and learns a single topographic map of a 2-D tissue slice in terms of a 1-D coordinate called isodepth, where all genes can be expressed as a function of this __isodepth__. Now, allowing features like metabolites to follow __mulitple distinct spatial patterns__, MET-MAP captures the feature topography by learning __k isodepths__, that smoothly vary across a tissue slice and capture spatial organizations of different groups of spatially variable features.
<p align="center">
<img src="plots/NNarchitecture.png" height=400/>
</p>

To support long-term reproducibility, this repository now includes the original source code required to run the metabolomics tutorials. This ensures that the MET-MAP tutorials remain functional and independent of future updates to the main Multi-GASTON package. This repository includes:
* **Original MET-MAP Source Code** (Preliminary version of Multi-GASTON)
* **Example datasets**
* **Neural network outputs**
* **Downstream metabolite analyses**

## Installation

1. **Option 1:** To install the preliminary version of the method used in this repository
   ```bash
   git clone https://github.com/raphael-group/MET-MAP.git
   cd MET-MAP
   ```
   
2. **Option 2:** To install the current version of Multi-GASTON, which is **compatible** with MET-MAP applications, please refer to Multi-GASTON installation at https://github.com/raphael-group/Multi-GASTON/tree/main. After installing Multi-GASTON package, simply activate the conda enviroment required for the jupyter notebooks.
```
conda activate multi_gaston_env
```

## Data availability
For liver and small intestine metabolomics data used in the paper, please refer to Figshare repositories:

Liver spatial metabolomics: https://doi.org/10.6084/m9.figshare.29318279.v1

Intestine spatial metabolomics: https://doi.org/10.6084/m9.figshare.29318342.v1

## Citation

Please cite our manuscript published at Nature if you use our method:

Samarah, L.Z., Zheng, C., Xing, X. et al. Spatial metabolic gradients in the liver and small intestine. Nature (2025). https://doi.org/10.1038/s41586-025-09616-5
