## Towards Sparsification of Graph Neural Networks

Comparison between SLR method and sparse training method on GNN dataset. 
<p align="center">
  <img src="imgs/sparsity.png" width="420">
  <br />
  <br />
  </p>

##### Train and prune: 
we use the SLR method for train and prune.

##### Sparse training: 
we follow same experiment setup as RigL paper, using weight magnitute for drop and weight gradient for grow. 

##### Train and prune: 

###### 1. Link prediction & SLR

Folder `SLR_Link_Pred` is the folder for SLR training for link prediction, which follows dense training -> reweight training -> sparse training procedure. 

###### 2. Node classification & SLR

Folder `SLR_Link_Pred` is the folder for SLR training for node classification, which follows dense training -> reweight training -> sparse training procedure. 
