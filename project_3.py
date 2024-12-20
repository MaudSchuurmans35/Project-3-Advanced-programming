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
from sklearn.model_selection import cross_val_score, cross_validate

def reading_data(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    path = os.path.join(script_dir, filename)  # Construct the full path to train.csv
    data = pd.read_csv(path)
    return data

def get_labels(data, manner='short'):
    all_labels=data['target_feature'] #get all labels 
    if manner == 'completely': #if manner = completely return all labels
        labels= all_labels
    else:
        labels = all_labels[0:200] #if manner = short return first 200
    return labels

def getting_descriptors(data,manner='short',index_smile=0):
    all_descriptors=[]
    all_fingerprints=[]
    if manner== 'completely':

        for smile in data.iloc[:,index_smile]:   #this is the correct one which loops over all the molecules but it takes 5 minutes to run
            molecule = MolFromSmiles(smile) #create the object molecule based on a smile
            vals = Descriptors.CalcMolDescriptors(molecule) #get the values of all the descripters from this molecule
            all_descriptors.append(vals) #create a list off all the dictionarys containing the descriptor information
    else:
        
        for i in range(200):  #we us this one because the other one takes to long to run
            smile=data.iloc[i,index_smile]
            molecule = MolFromSmiles(smile)
            vals = Descriptors.CalcMolDescriptors(molecule) #vals is a dictionary
            all_descriptors.append(vals)

            """deze 2 weggecommente lines zijn de fingerprints, deze kunnen misschien een beter model geven
            maar geeft wel heel veel hoofdpijn en maakt het ook veel ingewikkelder voor pca om goede dingen 
            te vinden want hierdoor komt er nBits aan features bij"""

            # morgan_fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024) #2048 
            # all_fingerprint.append(morgan_fp.ToBitString())

    all_descriptors_df = pd.DataFrame(all_descriptors) #turn the list of dictionarys into a dataframe
    return all_descriptors_df

def min_max_scaling_data(data_frame):
    scaler=MinMaxScaler() #create scaler object
    normalized_data_frame = pd.DataFrame(scaler.fit_transform(data_frame), columns=data_frame.columns) #normalize the data using minmax scaler
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

def pca(data, threshold_variance, plot=False):
    pca =PCA(n_components=threshold_variance)    #create pca object
    principal_components = pca.fit_transform(data) #perform pca
    if plot == True: #plot the pca plots if needed
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

        plt.show()
    data=pd.DataFrame(principal_components)
    return data, pca

def processing_train_data(feature_data, correlation_threshold,manner='short',plot=False):
    clean_data=rem_empty_columns(feature_data) #removing columns where all entries are the same
    scaled_data=min_max_scaling_data(clean_data) #scaling the data using a min-max scaler
    cleaner_data= rem_corr_features(scaled_data,correlation_threshold) #removing all highly correlated features
    return cleaner_data


def get_val_score(descriptors, target_feature):
    logistic_model = LogisticRegression(solver = 'lbfgs') # use this solver to make it faster
    scores = cross_val_score(logistic_model,descriptors,target_feature,cv=5,scoring='balanced_accuracy') #here the validation scores are calculated using 5 fold cross validation
    val_score=scores.mean() #take the mean off the 5 folds
    return val_score


def getting_cor_var(feature_data, labels,manner='short'): #this function tries to find the optimal values of the threshold for correlation and variance
    correlation = list(np.arange(0.8, 0.95, 0.025)) #try values for correlation from 0.8 to 0.95 with steps off 0.025
    variance=list(np.arange(0.8,0.95,0.025)) #try values for variance explained from 0.8 to 0.95 with steps off 0.025
    best_val_score=0
    best_corr=0
    best_var=0
    for i in correlation: #loop over all the different correlation values
        for j in variance: #loop over all the different variance values
            clean_data = processing_train_data(feature_data,i,manner) #process the data using selected correlation value
            x_data,pca_model=pca(clean_data,j) #perform pca using selected variance value
            val_score=get_val_score(x_data,labels) #get the validation scores
            if val_score > best_val_score: #if the new validation score is better then the old save the value together with the corresponding correlation and variance value
                best_val_score = val_score
                best_corr=round(i,3)
                best_var=round(j,2)
    print('the best balanced acc =',best_val_score)
    print('for correlation_max', best_corr, 'and variance explained', best_var)
    return best_corr, best_var

def train_model(descriptors, target_feature): #train the logistic regression model
    logistic_model = LogisticRegression(solver = 'lbfgs') # use this solver to make it faster
    logistic_model.fit(descriptors, target_feature)
    return logistic_model

def removing_features(train_data,test_data): #remove the features in the test data that are not used for pca in the training data
    columns_to_keep=list(train_data.columns) #make a list of all the columns that are used for pca
    test_df=test_data[columns_to_keep] #create a dataframe with all used features
    return test_df
           

def predict_from_smiles(test_data,train_data, logistic_model, pca_model,manner='short'):
    descriptors = getting_descriptors(test_data, manner, index_smile=1) # Extract descriptors for the SMILES data
    unique_ids=test_data['Unique_ID'].head(descriptors.shape[0]) # Extract descriptors for the SMILES data
    scaled_data = min_max_scaling_data(descriptors) # Clean and preprocess descriptors
    data=removing_features(train_data,scaled_data) #remove features not needed for pca transform 
    pca_data=pca_model.transform(data) #transforming the original descriptors to principle
    predictions = logistic_model.predict(pca_data) # Predict using the trained logistic regression model
    output_df = pd.DataFrame({'Unique_ID': unique_ids,'target_feature': predictions}) #turning the predictions into a dataframe
    return output_df

manner='completely'
train_data=reading_data('train.csv') #get the data out of the excel file
labels=get_labels(train_data,manner)
feature_data=getting_descriptors(train_data,manner) #getting the differend descriptors
print('now the training has started')
best_corr, best_var = getting_cor_var(feature_data,labels)
clean_data = processing_train_data(feature_data,best_corr,manner,False) #processing the data
X_data,pca_model =pca(clean_data, best_var, plot=False)
logistic_model1 = train_model(X_data, labels) #training the model

test_data=reading_data('test.csv') #get the test_data out of the excel file
print('starting with predicting outputs')
output_predictions = predict_from_smiles(test_data, clean_data, logistic_model1,pca_model,manner) #predict the labels for the test data
print(output_predictions)

# Save predictions to CSV
# script_dir = os.path.dirname(os.path.abspath(__file__)) 
# output_csv_path = os.path.join(script_dir, 'predicted_outcomes.csv')
# output_predictions.to_csv(output_csv_path, index=False)
