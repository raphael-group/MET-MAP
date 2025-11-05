# MET-MAP
Reproducibility repository for Multi-GASTON applied on spatial metabolomics data.

This repository provides two jupyter notebook tutorials on applying the deep learning model _Multi-GASTON_—-referred to here as _Metabolic Topography Mapper: MET-MAP_-—to spatial metabolomics data from murine liver and small intestine. To support reproducibility, we include example datasets, neural network outputs, and downstream metabolite analyses. Although these tutorials focus on metabolomics, MET-MAP is broadly applicable to any spatially-resolved data, enabling the recovery of tissue architecture and spatial patterns of feature variation across diverse organs. For more details about the model and analysis, please visit: https://github.com/raphael-group/Multi-GASTON and paper https://www.nature.com/articles/s41586-025-09616-5. 

<p align="center">
<img src="plots/liver_demo.png" height=400/>
<img src="plots/intestine_demo.png" height=400/>
</p>

## Installation
For enviroment setup, please refer to Multi-GASTON installation at https://github.com/raphael-group/Multi-GASTON/tree/main. After installing Multi-GASTON package, simply activate the conda enviroment required for the jupyter notebooks.

```
conda activate multi_gaston_env
```

## Reproducibility
For liver and small intestine metabolomics data used in the paper, please refer to Figshare repositories:

Liver spatial metabolomics: https://doi.org/10.6084/m9.figshare.29318279.v1

Intestine spatial metabolomics: https://doi.org/10.6084/m9.figshare.29318342.v1

## Citation

Please cite our manuscript published at Nature if you use our method:

Samarah, L.Z., Zheng, C., Xing, X. et al. Spatial metabolic gradients in the liver and small intestine. Nature (2025). https://doi.org/10.1038/s41586-025-09616-5
