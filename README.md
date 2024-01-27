# PUTransGCN: identiﬁcation of piRNA-disease associations based on attention encoding graph convolutional network and positive unlabelled learning

This repository contains the code for the paper: [PUTransGCN: identiﬁcation of piRNA-disease associations based on attention encoding graph convolutional network and positive unlabelled learning]().

PUTransGCN is a deep learning model for predicting piRNA-disease associations.

![Alt text](fig/flowchart.jpg?raw=true "PUTransGCN pipeline")


# Requirements

The code was tested in Python 3.10
Before running the code, please install packages below.
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
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
# Usage

1. Because GitHub has a file size limit for uploads, some files need to be upzip locally by yourself.
Unzip all the compressed files under the "/data/database" and /data" directory.
2. Run the `main.py` in each folder.
3. `result_compare.py` can consolidate the results of each model into one excel file for comparison.