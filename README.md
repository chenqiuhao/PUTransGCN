# PUTransGCN: identiﬁcation of piRNA-disease associations based on attention encoding graph convolutional network and positive unlabelled learning

This repository contains the code for the paper: [PUTransGCN: identiﬁcation of piRNA-disease associations based on attention encoding graph convolutional network and positive unlabelled learning]().

PUTransGCN is a deep learning model for predicting piRNA-disease associations.

![Alt text](fig/flowchart.jpg?raw=true "PUTransGCN pipeline")


# Getting Started

## Installation
Setup conda enviornment:
```
conda create -n PUTransGCN python=3.10 -y
conda activate PUTransGCN
```

Install required packages
```
pip install numpy==1.25.0
pip install scipy==1.11.1
pip install pandas==1.5.3
pip install openpyxl==3.0.10
pip install scikit-learn==1.2.2
pip install biopython==1.83
pip install obonet==1.0.0
pip install gensim==4.3.1
pip install tqdm==4.65.0
pip install jupyterlab==3.6.3
pip install matplotlib==3.8.2
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_cluster-1.6.2%2Bpt21cu118-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_sparse-0.6.18%2Bpt21cu118-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt21cu118-cp310-cp310-win_amd64.whl
pip install torch-geometric
```

# Prepare Datasets

Please download the raw data from https://zenodo.org/records/10686038
```
${DATA_ROOT}
├── comparison
│   ├── data (*MNDR v4.0 dataset*)
│   │   ├──adj.csv *(adjacent matrix)*
│   │   ├──p2p_smith.csv (*piRNA sequence similarity based on Simith-Waterman*)
│   │   ├──d2d_do.csv (*disease semantic similarity*)
│   │   ├──piRNA_seq.csv (*piRNA sequence information obtained from several databases*)
│   │   ├──doid.csv (*disease DO id*)
│   │   ├──gensim_feat_128.npy (*piRNA sequence feature*)
│   │   ├──database (*piRNA sequence databases*)
│   │   │  ├──Homo_sapiens.fasta
│   │   │  ├──hsa.v3.0.fa
│   │   │  ├──...
│   ├── data_piRDisease (*piRDisease v1.0 dataset*)
│   │   ├──...
│   ├── data_piRPheno (*piRPheno dataset*)
│   │   ├──...
│   ├── PUTransGCN_spy
│   │   ├──gen_rn_ij.py (*select reliable negative associations*)
│   │   ├──model.py (*model architecture*)
│   │   ├──utils.py (*utility function*)
│   │   ├──main.py
│   │   ├──PUTransGCN_spy.xlsx (*five-fold cross-validation result*)
│   ├── PUTransGCN_comb_5 (*pu_bagging + two_step + 5% spy*)
│   │   ├──model.py (*model architecture*)
│   │   ├──utils.py (*utility function*)
│   │   ├──main.py
│   │   ├──PUTransGCN_comb_5.xlsx (*five-fold cross-validation result*)
│   ├── ...
│   ├── result_compare.py (*script for comparison performance results*)
│   ├── result_compare.xlsx (*comparison results*)
│   ├── plot_curve.py (*script for ploting figure*)
├── case_study (*case study scripts*)
├── tutorial.ipynb (*tutorial for running and reproduce the results*)
├── PUTransGCN_model.pt (*trained weights for PUTransGCN network*)
```

# Training

To train models for cross-validation, please follow the following steps:
1. Run `main.py` in `PUTransGCN_comb_5` folder.
2. For other methods mentioned in the paper, run `main.py` in `ETGPDA`, `iPiDi-PUL`, `iPiDA-GCN`, `iPiDA-SWGCN`, `iPiDA-GBNN`，`PUTransGCN_all`, `PUTransGCN_pu_bagging`, `PUTransGCN_two_step`, `PUTransGCN_spy` folder.

# Comparison

Run `result_compare.py` to consolidate the results of each model into `result_compare.xlsx` for comparison.

# Notebook
`tutorial.ipynb` gives an instruction to run PUTransGCN and reproduce result.
