# NeuroPharmAI
![Logo](https://github.com/user-attachments/assets/1db15093-a1c8-4b28-b3a2-541af3e6f2b0)




## Goal 
The primary objective of our company is to develop a machine learning models which is capable to identify if a SMILE can bind to DRD3. 


## Input 

The code needs two input files. 

First is a csv file with the train data. This file should contain two columns, one with the SMILES and one with the corresponding "target_feature". 

Second is a csv file with the test data. This file should contain two columns with one the Unique_ID and one with the SMILES. 


## Requirements
- Python
- Matplotlib
- Sklearn
- Pandas
- Numpy
- Rdkit
- OS

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


## Output 

The code outputs a csv file. In file contains the following format:

Unique_ID, target_feature
1, 0
2, 1
3, 0 
etc. 

Where the 'Unique_ID' column corresponds to the number provided in the test dataset. The 'target_feature' corresponds to 0 or to 1 if a molecule is predicted to not bind or to bind to DRD3, respectively. 

