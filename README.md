# github repo for DGRCL

code for paper ’Dynamic Graph Representation with Contrastive Learning for Stock Movement Prediction Integrating Temporal Evolution and Static Relations‘ (DGRCL) submitted to AAAI 2025

![image](https://github.com/PEIYUNHUA/DGRCL/blob/main/%5Cfig_DGRCL.jpg)

## Environment Installation / Requirements
### Requirements
```
cppimport==22.8.2
joblib==1.3.2
logger==1.4
matplotlib==3.7.2
networkx==3.1
numpy==1.24.3
pandas==2.0.3
PyYAML==6.0.1
PyYAML==6.0.2
scikit_learn==1.3.0
scipy==1.14.0
torch==2.0.1+cu118
torch_geometric==2.3.1
torch_scatter==2.1.1+pt20cu118
torch_sparse==0.6.17+pt20cu118
```
To install a new environment with the required packages, run the following commands:

1. Create a new environment using conda 

(assuming miniconda installed on machine: https://docs.anaconda.com/miniconda/ ) 
```
conda create --name dgrcl python
```
2. Activate the environment
```
conda activate dgrcl
```
3. Install the required packages
 
_Using the requirements.txt file_:
```
pip install -r requirements.txt
```
_Or using the install.sh script_:
```
install.sh
```

### Data
The project's data is from "Temporal Relational Ranking for Stock Prediction" https://github.com/fulifeng/Temporal_Relational_Stock_Ranking.

### process_data

- **\preprocessing\gen_data.py ⟶ generate structured data**

- **\preprocessing\gen_graph.py ⟶ get Adjacency Matrix, X and Y:**

### run_prediction

- **train.py:**

  Adjust hyper parameters in \experiments\parameters_dgrcl.yaml.


