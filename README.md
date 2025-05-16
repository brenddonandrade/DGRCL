# github repo for DGRCL

code for the paper ’Dynamic Graph Representation with Contrastive Learning for Financial Market Prediction: Integrating Temporal Evolution and Static Relations‘ (DGRCL) accepted by ICAART 2025 (https://icaart.scitevents.org/Home.aspx)

![image](https://github.com/PEIYUNHUA/DGRCL/blob/main/fig_DGRCL.jpg)

## Environment Installation / Requirements
### Requirements

To install all requirements, use:
```
# to create the environment
conda env create -f environment.yml

# lib to cuda
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install torch-geometric==2.3.1 \
  torch-scatter==2.1.1 \
  torch-sparse==0.6.17 \
  torch-cluster==1.6.1 \
  torch-spline-conv==1.2.2 \
  -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```
To install a new environment with the required packages, run the following commands:

1. Create a new environment using conda 

(assuming miniconda installed on machine: https://docs.anaconda.com/miniconda/ ) 
```
conda env create -f environment.yml
```

2. Activate the environment
```
conda activate env_dgrcl
```

### Data
The project's data is from "Temporal Relational Ranking for Stock Prediction" https://github.com/fulifeng/Temporal_Relational_Stock_Ranking.

To get, use:
```
# Create directory to data
mkdir data/google_finance

# get data 
git clone https://github.com/fulifeng/Temporal_Relational_Stock_Ranking.git
cp Temporal_Relational_Stock_Ranking/data/google_finance/* data/google_finance

# remove repository used to fetch data
rm -rf Temporal_Relational_Stock_Ranking
```

### Processing the data

To process data and generate the data to framework, user:
```
# Remember, activate the enviroment
conda activate env_dgrcl

# Then, use the script:
python -u preprocessing/gen_data.py 2>&1 | tee -a gen_data.log

# this do that any error (2>&1) or standard output going to gen_data.log
```

- **\preprocessing\gen_data.py ⟶ generate structured data**

```
# Next, this script going to generate the graph by Adjacency Matrix,
python -u preprocessing/gen_graph.py 2>&1 | tee -a gen_graph.log
```

- **\preprocessing\gen_graph.py ⟶ get Adjacency Matrix, X and Y:**

### run_prediction

- **train.py:**

```
# Finaly, this script going to generate the graph by Adjacency Matrix,
python -u train.py 2>&1 | tee -a train.log
```


  Adjust hyper parameters in \experiments\parameters_dgrcl.yaml.



