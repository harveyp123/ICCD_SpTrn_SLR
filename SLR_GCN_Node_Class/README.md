
#### This directory is for SLR training over 1 node classification dataset: brain

Steps to reproduce the results:

##### 1. Install the environment: 

`pip install -r requirements.txt`
<br />

##### 2. <a href="https://drive.google.com/file/d/1jlfQHiz1slZ6mDeXgHABoIGEZffa--Ln/view?usp=sharing" target="_blank">Download</a> the dataset,  
unzip the file and put it into the `dataset` folder

##### 3. Run the dense training:

Go to the main/SLR_Link_Pred/bash_scripts folder, than run the bash scripts for a task. 

For example, run the dense training for ia-email dataset: 

`cd /main/SLR_Node_Class/bash_scripts/brain_dense`

Then modify the 1st line of the 'run_dense_training.sh' to `cd path/to/home/main/SLR_Node_Class/`. Change `path/to/home` to your own downloading folder.
Also change the `--running-dir /home/hop20001/ADMM_SLR_GNN_Node_class` to `--running-dir path/to/home/main/SLR_Node_Class/` after `python main_dense.py`. Then run:

`bash run_dense_training.sh`
<br />


##### 4. Run the SLR based reweighted training and retraining:

Go to the main/SLR_Link_Pred/bash_scripts folder, than run the bash scripts for a task. 

For example, run the SLR training for ia-email dataset: 

`cd main/SLR_Link_Pred/bash_scripts/ia_email_dense`

Then modify the 1st line of the 'run_slr_training.sh' to `cd path/to/home/main/SLR_Link_Pred/`. Change `path/to/home` to your own downloading folder. Also change the `--running-dir /home/hop20001/ADMM_SLR_GNN_Node_class` to `--running-dir path/to/home/main/SLR_Node_Class/` after `python main_admm.py`. Then run:

`bash run_slr_training.sh`


##### 5. (optional) Run the ADMM based reweighted training and retraining:

Go to the main/SLR_Link_Pred/bash_scripts folder, than run the bash scripts for a task. 

For example, run the ADMM training for ia-email dataset: 

`cd main/SLR_Link_Pred/bash_scripts/ia_email_dense`

Then modify the 1st line of the 'run_admm_training.sh' to `cd path/to/home/main/SLR_Link_Pred/`. Change `path/to/home` to your own downloading folder. Also change the `--running-dir /home/hop20001/ADMM_SLR_GNN_Node_class` to `--running-dir path/to/home/main/SLR_Node_Class/` after `python main_admm.py`. Then run:

`bash run_slr_training.sh`

