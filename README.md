# NeuroPharmAI
![Logo](https://github.com/user-attachments/assets/1db15093-a1c8-4b28-b3a2-541af3e6f2b0)




## Goal 
The primary objective of our company is to develop a machine learning models which is capable to identify if a SMILE can bind to Dopamine Receptor D3 (DRD3). DRD3 is a G-protein-coupled receptor primarily found in brain regions that regulate mood and motor control. It plays a significant role in neurological and psychiatric conditions such as Parkinson’s disease, and addiction. Identifying molecules that can bind to DRD3 can aid in developing therapies targeting these disorders.


## Input 

The code needs two input files. 

First is a csv file with the train data. This file should contain two columns, one with the SMILES and one with the corresponding "target_feature". 

```
SMILES_canonical,"target_feature"
CCOC(=O)CC1ON(C)C(=O)N1c1ccc([N+](=O)[O-])cc1,"0"
Cc1nc(C)c(-c2nnc(SCCCN3CCC4(CCc5cc(Cl)ccc54)C3)n2C)s1,"1"
COC(=O)c1c(NC(=O)c2nc(SCc3cccc(C)c3)ncc2Cl)sc2c1CCCC2,"0"
Cc1c(Cl)cccc1-n1nnc(-c2nc(-c3ccc4c(c3)OCO4)no2)c1C,"0"
etc.
```

Second is a csv file with the test data. This file should contain two columns with one the Unique_ID and one with the SMILES. 

```
Unique_ID,"SMILES_canonical"
1,"COc1cccc(C(=O)N2CCCC(CCC(=O)NCc3ccc(F)c(F)c3)C2)c1"
2,"OCc1ccc2c(CN3CCN(c4ccc(Cl)cc4)CC3)cnn2c1"
3,"CCN1CCc2c(sc(NC(=O)COc3ccccc3)c2C(=O)OC)C1"
etc.
```



## Requirements
- Python
- Matplotlib
- Sklearn
- Pandas
- Numpy
- Rdkit
- OS

## Usage
The final code is the "final main assignment.py" file in this resporatory. 
To use the code you open the file "run file.py". By opening and running this file, the code is runned. This file contains the following code:
```
from final_main_assignment import run_code
run_code(manner = 'short', fingerprintcount = 4096, augment_data_x_times = 5, write_to_file = False, train_nn_model = False)
```
The manner, fingerprintcount and augment data x times can be changed. Additionally, the predictions can also be written to a file and if desirable, a Neural Network can also be trained. 



## Key Functions
All the different functions, with a small explanation, present in our code are stated below. 


reading_data(filename): Load dataset from a CSV file.  
augment_data(data, size): Augment dataset by duplicating data.  
get_labels(data, manner): Extract labels from the dataset.  
getting_descriptors(data, manner, index_smile): Generate molecular descriptors and fingerprints.    
min_max_scaling_data(data_frame): Normalize features using MinMaxScaler.  
rem_empty_columns(data): Remove columns with constant values.  
rem_corr_features(data, threshold): Remove highly correlated features.  
pca(data, threshold_variance, plot): Perform PCA for dimensionality reduction.  
processing_train_data(feature_data, correlation_threshold, manner, plot): Preprocess training data.  
get_val_score(descriptors, target_feature): Evaluate model performance using cross-validation.  
getting_cor_var(feature_data, labels, manner): Optimize correlation and variance thresholds.  
train_logistic_model(descriptors, target_feature): Train a logistic regression model.  
removing_features(train_data, test_data): Ensure test data matches training feature space.  
preparing_test_data(test_data, train_data, pca_model, manner): Preprocess test data.  
predict_from_smiles(test_data, trained_model): Make predictions using the trained model. 
run_code(manner, fingerprintcount, augment_data_x_times, write_to_file, train_nn_model): Running this function, the code is runned and calls all the other functions

## Output 

The code outputs a csv file. In file contains the following format:
```
Unique_ID, target_feature
1, 0
2, 1
3, 0 
etc. 
```

Where the 'Unique_ID' column corresponds to the number provided in the test dataset. The 'target_feature' corresponds to 0 or to 1 if a molecule is predicted to not bind or to bind to DRD3, respectively. 

