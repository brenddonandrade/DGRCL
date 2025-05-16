# github repo for DGRCL

code for the paper ’Dynamic Graph Representation with Contrastive Learning for Financial Market Prediction: Integrating Temporal Evolution and Static Relations‘ (DGRCL) accepted by ICAART 2025 (https://icaart.scitevents.org/Home.aspx)

![image](https://github.com/PEIYUNHUA/DGRCL/blob/main/fig_DGRCL.jpg)

## Environment Installation / Requirements
### Requirements

To install all requirements, use:
```
conda env create -f environment.yml
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
python -u preprocessing/gen_data.py > gen_data.log 2>&1 

# this do that any error (2>&1) or standard output going to gen_data.log
```
- **\preprocessing\gen_data.py ⟶ generate structured data**

```
# Next, this script going to generate the graph by Adjacency Matrix,
python -u preprocessing/gen_graph.py > gen_graph.log 2>&1 

```
- **\preprocessing\gen_graph.py ⟶ get Adjacency Matrix, X and Y:**

### run_prediction

- **train.py:**

```
# Finaly, this script going to generate the graph by Adjacency Matrix,
python -u train.py > train.log 2>&1
```


  Adjust hyper parameters in \experiments\parameters_dgrcl.yaml.



