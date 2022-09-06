
#### This directory is for Sparse training over GCN on 3 dataset: Cora, Pubmed, and CiteSeer

Steps to reproduce the results:

##### 1. Install the environment: 

Using the conda to create a new environment and activate it. 
Then run following code to install pytorch 1.12.1 on cuda 10.2: 

`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

Then install pyg:

`pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu102.html`

`pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu102.html`

`pip install torch-geometric`

Install other necessary requirement:
`pip install -r requirements.txt`
<br />


##### 2. Run the sparse training:

`bash run.sh`
<br />


