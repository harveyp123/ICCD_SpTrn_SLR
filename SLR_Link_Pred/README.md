#### This directory is for SLR training over 3 link prediction dataset: ia-email, wiki-talk, stackoverflow

Steps to reproduce the results:

##### 1. Install the environment: 

`pip install -r requirements.txt`

<br />


##### 2. Run the dense training:

Go to the main/SLR_Link_Pred/bash_scripts folder, than run the bash scripts for a task. 

For example, run the dense training for ia-email dataset: 

`cd main/SLR_Link_Pred/bash_scripts/ia_email_dense`

Then modify the 1st line of the 'run_dense_training.sh' to `cd path/to/home/main/SLR_Link_Pred/`. Change `path/to/home` to your own downloading folder.
Also change the `--running-dir /home/hop20001/ADMM_SLR_GNN` to `--running-dir path/to/home/main/SLR_Link_Pred/` after `python main_dense.py`. Then run:

`bash run_dense_training.sh`

<br />


##### 3. Run the SLR training:

Go to the main/SLR_Link_Pred/bash_scripts folder, than run the bash scripts for a task. 

For example, run the SLR training for ia-email dataset: 

`cd main/SLR_Link_Pred/bash_scripts/ia_email_dense`

Then modify the 1st line of the 'run_slr_training.sh' to `cd path/to/home/main/SLR_Link_Pred/`. Change `path/to/home` to your own downloading folder. Also add the `--running-dir path/to/home/main/SLR_Link_Pred/` after `python main_dense.py`Then run:

`bash run_slr_training.sh`
