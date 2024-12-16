import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors 
from rdkit.Chem import MolFromSmiles 
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

def reading_data(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    path = os.path.join(script_dir, filename)  # Construct the full path to train.csv
    data = pd.read_csv(path)
    return data
data=reading_data('train.csv')

def getting_descriptors(data,maner='short',index=0):
    all_descriptors=[]
    all_fingerprints=[]
    if maner== 'completely':

        for smile in data.iloc[:,index]:   #this is the correct one which loops over all the molecules but it takes 5 minutes to run
            molecule = MolFromSmiles(smile)
            vals = Descriptors.CalcMolDescriptors(molecule) #vals is a dictionary
            all_descriptors.append(vals)
    else:
        
        for i in range(10):  #we us this one because the other one takes to long to run
            smile=data.iloc[i,0]
            molecule = MolFromSmiles(smile)
            vals = Descriptors.CalcMolDescriptors(molecule) #vals is a dictionary
            all_descriptors.append(vals)

            """deze 2 weggecommente lines zijn de fingerprints, deze kunnen misschien een beter model geven
            maar geeft wel heel veel hoofdpijn en maakt het ook veel ingewikkelder voor pca om goede dingen 
            te vinden want hierdoor komt er nBits aan features bij"""

            # morgan_fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024) #2048 
            # all_fingerprint.append(morgan_fp.ToBitString())

    all_descriptors_df = pd.DataFrame(all_descriptors) #this is a dataframe whit all different features
    return all_descriptors_df


def min_max_scaling_data(data_frame):
    scaler=MinMaxScaler()
    normalized_data_frame = pd.DataFrame(scaler.fit_transform(data_frame), columns=data_frame.columns)
    return normalized_data_frame

#if the value of a feature is the same everywhere we can throw this feature away
def rem_empty_columns(data):
    for column in data.columns:
        if len(set(data[column])) == 1:  #turn the values of the column in a set, if the length is 1 all entries are the same
            #scaled_data=data.drop(column, axis=1) #drop the corresponding rows
            new_data=data.drop(column, axis=1, inplace=False) #drop the corresponding rows
    #return scaled_data
    return new_data

def rem_corr_features(data,threshold):
    cor_matrix=data.corr()
    columns_to_drop=set()
    for index, row in cor_matrix.iterrows(): #looping over all the rows in the correlation matrix
        if index in columns_to_drop: #checking if the row is already in the set columns_to_drop it is not needed to check all the correlations again
            continue
        for col in cor_matrix.columns: #loop over all the columns
            if cor_matrix.at[index,col] > threshold and index != col: #select the corresponding values and check if they are above the threshold and the 2 features are not the same
                columns_to_drop.add(col) #add the corresponding column to the set which contains all features which have high variance with another feature

    new_data=data.drop(columns_to_drop, axis=1, inplace=False) #removing all highly correlated features
    return new_data



#extracting information
feature_data=getting_descriptors(data,'completel') #extracting all descriptors
#cleaning data
clean_data=rem_empty_columns(feature_data) #removing columns where all entries are the same
scaled_data=min_max_scaling_data(clean_data) #scaling the data using a min-max scaler
cleaner_data= rem_corr_features(scaled_data,0.9) #removing all highly correlated features
print(list(cleaner_data.columns))

def pca(data, threshold_variance):
    pca =PCA(n_components=threshold_variance)    #waarom bij 
    principal_components = pca.fit_transform(data)
    loadings = pca.components_

    # print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    # print("Cumulative Explained Variance:", np.cumsum(pca.explained_variance_ratio_))
    # Plot 1: Cumulative Explained Variance Ratio
    plt.figure(1)
    plt.bar(list(range(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1)), np.cumsum(pca.explained_variance_ratio_), color='skyblue', edgecolor='black')
    plt.title('Cumulative Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(['Cumulative Explained Variance'], loc='best')

    # Plot 2: Explained Variance Ratio for Each Principal Component
    plt.figure(2)
    plt.plot(list(range(1, len(pca.explained_variance_ratio_) + 1)), pca.explained_variance_ratio_, marker='o', color='orange', linewidth=2, markersize=6)
    plt.title('Explained Variance Ratio for Each Principal Component')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(['Explained Variance Ratio'], loc='best')

    # Show both plots
    # plt.show()
    # print(len(principal_components[0]))
    data=pd.DataFrame(principal_components)
    # print(data.head)
    return data

final_data = pca(cleaner_data,None)




def train_model(descriptors, target_feature):
    logistic_model = LogisticRegression(solver = 'lbfgs') # use this solver to make it faster
    logistic_model.fit(descriptors, target_feature)
    return logistic_model
    # performing cross validation with different 
    # cv_results = cross_validate(logisticRegr, descriptors, target_feature['target_feature'], cv=5, scoring=['balanced_accuracy'])

    # # Print the results
    # print("Cross-validation results:", cv_results)
    # print("Mean Balanced Accuracy:", cv_results['test_balanced_accuracy'].mean())

# logisticRegression(final_data, data)

def predict_from_smiles(input_data, logistic_model, threshold_variance):
    """
    Function to predict outcomes based on SMILES data in an input CSV.
    
    Args:
        input_csv_path (str): Path to the input CSV containing 'unique_id' and 'smile'.
        logistic_model (LogisticRegression): Trained logistic regression model.
        threshold_variance (float): PCA threshold variance to use for dimensionality reduction.

    Returns:
        pd.DataFrame: DataFrame containing 'unique_id' and predicted 'target_feature'.
    """
    # Load the input CSV
    # input_data = pd.read_csv(input_csv_path)
    
    # Extract descriptors for the SMILES data
    descriptors = getting_descriptors(input_data, maner='completely', index=1)
    print(descriptors.head)

    # Clean and preprocess descriptors
    scaled_data = min_max_scaling_data(descriptors)
    



    # Predict using the trained logistic regression model
    predictions = logistic_model.predict(final_data)

    # Prepare the output DataFrame
    output_df = pd.DataFrame({
        'unique_id': input_data['unique_id'],
        'target_feature': predictions
    })

    return output_df

logistic_model1 = train_model(final_data, data['target_feature'])

test_data=reading_data('test.csv')
output_predictions = predict_from_smiles(test_data, logistic_model1, threshold_variance=None)

# Save predictions to CSV
script_dir = os.path.dirname(os.path.abspath(__file__)) 
output_csv_path = os.path.join(script_dir, 'predicted_outcomes.csv')
output_predictions.to_csv(output_csv_path, index=False)
