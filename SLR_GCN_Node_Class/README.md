
#### This directory is for SLR training over GCN on 3 dataset: Cora, Pubmed, and CiteSeer

Steps to reproduce the results:

##### 1. Install the environment: 

Using the conda to create a new environment and activate it. 
Then run following code to install pytorch on cuda 11.6: 
`conda install pytorch=1.12.0 torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge`

Then install pyg:

`pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html`

`pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html`

`pip install torch-geometric`

Install other necessary requirement:
`pip install -r requirements.txt`
<br />


##### 2. Run the dense training:

Go to the main/SLR_GCN_Node_Class/bash_scripts folder, than run the bash scripts for a task. 

For example, run the dense training for ia-email dataset: 

`cd /main/SLR_GCN_Node_Class/bash_scripts/Cora`

Then modify the 1st line of the 'run_dense_training.sh' to `cd path/to/home/main/SLR_GCN_Node_Class/`. Change `path/to/home` to your own downloading folder.
Also change the `--running-dir /home/hop20001/SLR_GCN_Node_Class` to `--running-dir path/to/home/main/SLR_GCN_Node_Class/` after `python main_dense.py`. Then run:

`bash run_dense_training.sh`
<br />


##### 3. Run the SLR based reweighted training and retraining:

Go to the main/SLR_GCN_Node_Class/bash_scripts folder, than run the bash scripts for a task. (Need to run 2th step to get a dense model first)

For example, run the SLR training for Cora dataset: 

`cd main/SLR_GCN_Node_Class/bash_scripts/Cora`

Then modify the 1st line of the 'run_slr_training.sh' to `cd path/to/home/main/SLR_GCN_Node_Class/`. Change `path/to/home` to your own downloading folder. Also change the `--running-dir /home/hop20001/ADMM_SLR_GNN_Node_class` to `--running-dir path/to/home/main/SLR_GCN_Node_Class/` after `python main_admm.py`. Then run:

`bash run_slr_training.sh`


##### 4. (optional) Run the ADMM based reweighted training and retraining:

Modify the SLR training script, change `export OPT=savlr` to `export OPT=admm`, and follow the same step as step 3 to run ADMM training. 

