Scenario Requirements: torch==1.9.0 python==3.7.16 dgl==0.6.1 numpy==1.20.0 pandas==1.3.5 rdkit==2020.9.5.2 torch-geometric==2.2.0


Run: 1. Run data_pre_k_fold.py to generate five alternative cross data, python data_pre_k_fold.py. 2. Run train.py to train MCF-DDI Python train.py.
If you want to regenerate a drug-target based similarity matrix, the file under the get_similarity_matric directory is executed.