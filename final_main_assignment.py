#%% 
import time
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


def reading_data(filename):
    print("entered reading data")
    """function reads data from file and returns the data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    path = os.path.join(script_dir, filename)  # Construct the full path to train.csv
    data = pd.read_csv(path)
    return data

def augment_data(data, size):
    """Multiplies the dataset. The function returns the multiplied dataset"""
    print("Entered augment data")
    augmented_dataset = data
    for i in range(size-1):
        augmented_dataset = pd.concat([augmented_dataset, data], ignore_index = True)
        print(len(augmented_dataset))
    return augmented_dataset


def get_labels(data, manner='short'):
    print("entered get labels")
    """function gets all labels from the data and returns the labels if manner is completely"""
    all_labels=data['target_feature'] #get all labels 
    if manner == 'completely': #if manner = completely return all labels
        labels= all_labels
    else:
        labels = all_labels[0:200] #if manner = short return first 200
    return labels

def getting_descriptors(data,manner='short',index_smile=0,fingerprintcount=1024):
    print("entered getting descriptors")
    """The function gets all descriptors of the SMILES.
    The function also appends the Morgan Fingerprints asa Bit Vector in a list. The fingerprints
    are put into a dataframe and the complete dataframe with all descriptors and the fingerprints is returned"""
    descriptors_start_time=time.time()
    all_descriptors=[]
    all_fingerprints=[]
    if manner== 'completely':

        for smile in data.iloc[:,index_smile]:   #this is the correct one which loops over all the molecules but it takes 5 minutes to run
            molecule = MolFromSmiles(smile) #create the object molecule based on a smile
            vals = Descriptors.CalcMolDescriptors(molecule) #get the values of all the descripters from this molecule
            all_descriptors.append(vals) #create a list off all the dictionarys containing the descriptor information
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=fingerprintcount) #2048 
            all_fingerprints.append(np.array(morgan_fp))
    else:
        
        for i in range(200):  #we us this one because the other one takes to long to run
            smile=data.iloc[i,index_smile]
            molecule = MolFromSmiles(smile)
            vals = Descriptors.CalcMolDescriptors(molecule) #vals is a dictionary
            all_descriptors.append(vals)
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=fingerprintcount) #2048 
            all_fingerprints.append(np.array(morgan_fp))
    # turning the fingerprints into a dataframe where every column is a different bit
    df_fingerprints=pd.DataFrame(all_fingerprints, columns = [f'Bit_{i}' for i in range(len(all_fingerprints[0]))])

    all_descriptors_df = pd.DataFrame(all_descriptors) #turn the list of dictionarys into a dataframe

    complete_df=pd.concat([all_descriptors_df, df_fingerprints], axis=1) #creating 1 dataframe out off the descriptors and the fingerprints
    return complete_df

def min_max_scaling_data(data_frame):
    """The function normalizes the dataframe and returns the normalized dataframe"""
    print("entered min max scaling")
    scaler=MinMaxScaler() #create scaler object
    normalized_data_frame = pd.DataFrame(scaler.fit_transform(data_frame), columns=data_frame.columns) #normalize the data using minmax scaler
    return normalized_data_frame

#if the value of a feature is the same everywhere we can throw this feature away
def rem_empty_columns(data):
    """Function removes columns in a dataset that have the same value in every row. Returns the updated data"""
    print("entered rem empty columns")
    for column in data.columns:
        if len(set(data[column])) == 1:  #turn the values of the column in a set, if the length is 1 all entries are the same
            #scaled_data=data.drop(column, axis=1) #drop the corresponding rows
            new_data=data.drop(column, axis=1, inplace=False) #drop the corresponding rows
    #return scaled_data
    return new_data

def rem_corr_features(data, threshold):
    """The function removes higly correlated features above a certain threshold.
    Returns updated dataframe without the highly correlated features"""
    cor_matrix = data.corr().abs()  # Absolute correlation values
    upper = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return data.drop(to_drop, axis=1)



def pca(data, threshold_variance, plot=False):
    """Function applies Principal Component Analysis to reduce the dimensionality
    of a dataset while retaining as much variance as possible, based on given threshold.
    Functions plots the Cumulative Explained Variance Ratio
    and Explained Variance Ratio for each principal component if plot is True
    The reduced-dimensionality dataset is converted back to a dataframe and returned"""
    print("entered pca")
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
    """function preprocesses the data by removing empty columns, scaling the data and removing higly correlated features"""
    print("entered processing train data")
    clean_data=rem_empty_columns(feature_data) #removing columns where all entries are the same
    scaled_data=min_max_scaling_data(clean_data) #scaling the data using a min-max scaler
    cleaner_data= rem_corr_features(scaled_data,correlation_threshold) #removing all highly correlated features
    return cleaner_data


def get_val_score(descriptors, target_feature, ML_model):
    """function returns validation score using k-fold cross validation """
    print("entered get val score")
    scores = cross_val_score(ML_model,descriptors,target_feature,cv=5,scoring='balanced_accuracy') #here the validation scores are calculated using 5 fold cross validation
    val_score=scores.mean() #take the mean off the 5 folds
    return val_score


def getting_cor_var(feature_data, labels, ML_model,manner='short'): 
    """function tries to find the optimal values of the threshold for correlation and variance and returns them"""
    print("entered getting cor var")
    correlation = list(np.arange(0.8, 0.95, 0.025)) #try values for correlation from 0.8 to 0.95 with steps of 0.025
    variance=list(np.arange(0.8,0.95,0.025)) #try values for variance explained from 0.8 to 0.95 with steps of 0.025
    best_val_score=0
    best_corr=0
    best_var=0
    for i in correlation: #loop over all the different correlation values
        for j in variance: #loop over all the different variance values
            clean_data = processing_train_data(feature_data,i,manner) #process the data using selected correlation value
            x_data,pca_model=pca(clean_data,j) #perform pca using selected variance value
            val_score=get_val_score(x_data,labels,ML_model) #get the validation scores
            if val_score > best_val_score: #if the new validation score is better then the old save the value together with the corresponding correlation and variance value
                best_val_score = val_score
                best_corr=round(i,3)
                best_var=round(j,3)
    print('the best balanced acc =',best_val_score)
    print('for correlation_max', best_corr, 'and variance explained', best_var)
    return best_corr, best_var

def train_logistic_model(descriptors, target_feature): 
    """function trains the logistic regression model and returns this model"""
    print("entered train model")
    logistic_model = LogisticRegression(solver = 'lbfgs') # use this solver to make it faster
    logistic_model.fit(descriptors, target_feature)
    return logistic_model

def train_svm_model(descriptors, target_feature):
    """function trains support vector machine.with a rbf kernel and specified hyperparameters. The trained model is returned. """
    svm_model=SVC(kernel='rbf', C=2.5, gamma='auto', random_state=42) #gamma is a hyperparameter, scale automaticly finds a good gamma based on feature space and variance rbf kernel is a radial basis function
    svm_model.fit(descriptors, target_feature)
    return svm_model


def balanced_accuracy(y_true, y_pred):
    """calculates the balanced accuracy for binary classification
    based on sensitivity and specifity of the predictions.
    The balanced accuracy is returned"""
    # Threshold predictions at 0.5 (for binary classification)
    y_pred_classes = K.cast(K.greater(y_pred, 0.5), dtype='float32')
    
    # True positives, false negatives, true negatives, false positives
    TP = K.sum(y_true * y_pred_classes)
    FN = K.sum(y_true * (1 - y_pred_classes))
    TN = K.sum((1 - y_true) * (1 - y_pred_classes))
    FP = K.sum((1 - y_true) * y_pred_classes)
    
    # Sensitivity (Recall for Class 1)
    sensitivity = TP / (TP + FN + K.epsilon())
    
    # Specificity (Recall for Class 0)
    specificity = TN / (TN + FP + K.epsilon())
    
    # Balanced Accuracy
    balanced_acc = (sensitivity + specificity) / 2
    return balanced_acc

def create_nn_model(X_train,learningrate=0.001):
    """function creates neural network model with layers and dropout, the model is compiled and returned. """
    model = Sequential()

    model.add(Dense(2*X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(X_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(round(X_train.shape[1]/2),activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(round(X_train.shape[1]/6),activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(round(X_train.shape[1]/12),activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=learningrate), loss='binary_crossentropy', metrics=['accuracy',balanced_accuracy])
    return model

def training_nn_model(X_train, labels, model,n_epochs=500,n_batchsize=64):
    """function trains the Neural network. The dataset is splitted in training and validation dataset.
    the history of the model is returned. """
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, labels, test_size=0.2, random_state=42)
    history = model.fit(X_train_split,y_train_split,validation_data=(X_val, y_val),epochs=n_epochs,batch_size=n_batchsize,verbose=1)
    return history

def plot_balanced_accuracy(history):
    """The validation loss and training loss is plotted for each epoch"""
    # Get the training and validation balanced accuracy from the history object
    train_balanced_acc = history.history.get('balanced_accuracy', [])
    val_balanced_acc = history.history.get('val_balanced_accuracy', [])
    val_loss=history.history.get('val_loss', [])
    train_loss=history.history.get('loss')

    plt.figure(figsize=(10, 6))
    plt.plot(val_loss, label='validation loss')
    plt.plot(train_loss, label='train loss')
    plt.legend()
    plt.show()
    
    # Plot the balanced accuracy for training and validation
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_balanced_acc, label='Training Balanced Accuracy')
    # plt.plot(val_balanced_acc, label='Validation Balanced Accuracy')
    # plt.title('Balanced Accuracy over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Balanced Accuracy')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    

def removing_features(train_data,test_data): 
    """function removes the features in the test data that are not used for pca in the training data and returns test dataframe"""
    print("entered removing features")
    columns_to_keep=list(train_data.columns) #make a list of all the columns that are used for pca
    test_df=test_data[columns_to_keep] #create a dataframe with all used features
    return test_df

def preparing_test_data(test_data,train_data,pca_model,manner='short',fingerprintcount=1024):
    """function prepares the test data by getting the descriptors, scaling the data and removing the features that 
    are not needed for pca.  The original descriptors are transformed to the principle components
    and the pca-transformed test data is returned with unique IDs as the first column"""
    descriptors = getting_descriptors(test_data, manner, index_smile=1, fingerprintcount=fingerprintcount) # Extract descriptors for the SMILES data
    unique_ids=test_data['Unique_ID'].head(descriptors.shape[0]) # Extract unique ids for the test data
    scaled_data = min_max_scaling_data(descriptors) # scale the test data
    data=removing_features(train_data,scaled_data) #remove features not needed for pca transform 
    pca_data=pca_model.transform(data) #transforming the original descriptors to principle components
    pca_columns = [i+1 for i in range(pca_data.shape[1])] # creating a list with the pca column names
    test_data_pca = pd.DataFrame(pca_data, columns=pca_columns) #turning the pca_data which is a np array into a dataframe
    test_data_pca.insert(0, 'Unique_ID', unique_ids.values) #adding the unique ids as the first column
    return test_data_pca

def predict_test_data(test_data,trained_model):
    """The function predicts the labels for the dataset based on the trained model. 
    The output is put into a dataframe with the unique IDs and this is returned"""
    unique_ids=test_data['Unique_ID'] #saving the unique ids
    predict_data=test_data.drop(columns=['Unique_ID'])
    #print(test_data.head)
    predictions = trained_model.predict(predict_data) #predicting the labels
    output_df = pd.DataFrame({'Unique_ID': unique_ids,'target_feature': predictions}) #turning the predictions into a dataframe with corresponding ids
    return output_df



def run_code(manner = 'short', fingerprintcount = 4096, augment_data_x_times = 5, write_to_file = False, train_nn_model = False):
    """Functions runs the code. It extracts and prepares the data, creates different ML models and cleans the data.
    It also gets the validation scores for the different models. Trains the models to predict the labels. The test data is prepared and predictions can be saved 
    to CSV file after the predictions are made. """
    ###extracting and preparing information
    train_data=reading_data('train.csv') #get the data out of the excel file
    augmented_training_data = augment_data(train_data, augment_data_x_times)
    labels=get_labels(augmented_training_data,manner)
    descriptors_start_time=time.time()
    feature_data=getting_descriptors(augmented_training_data,manner,0,fingerprintcount) #getting the differend descriptors ongeveer 2 minuten om standaard dataset de features eruit te halen en iets van 4 voor de fingerprints
    descriptors_end_time=time.time()

    ###creating different ml models
    logistic_model = LogisticRegression(solver = 'lbfgs')
    svm_model=SVC(kernel='rbf', C=2.5, gamma='auto', random_state=42)

    descriptors_time=descriptors_end_time-descriptors_start_time
    ###start with cleaning and preparing the data
    print('now the training has started')

    if manner == 'completely':
        best_corr_logistic, best_var_logistic = getting_cor_var(feature_data,labels,logistic_model,manner)
        best_corr_svm, best_var_svm = getting_cor_var(feature_data,labels,svm_model,manner)
    else: 
        best_corr = 0.825
        best_var = 0.875
    
    cleaning_start_time=time.time()
    clean_data = processing_train_data(feature_data,best_corr,manner,False) #processing the data
    #clean_data_svm = processing_train_data(feature_data,best_corr_svm,manner,False)
    # cleaner_data=rem_empty_columns(feature_data) #removing columns where all entries are the same
    # scaled_data=min_max_scaling_data(cleaner_data) #scaling the data using a min-max scaler
    # clean_data= rem_corr_features(scaled_data,best_corr) #removing all highly correlated features

    cleaning_end_time=time.time()
    pca_start_time=time.time()

    X_data,pca_model =pca(clean_data, best_var, plot=False)
    #X_data_svm,pca_model_svm =pca(clean_data_svm, best_var_svm, plot=False)
    pca_end_time=time.time()

    if train_nn_model == True:
        nn_model=create_nn_model(X_data)
        model_history=training_nn_model(X_data,labels,nn_model,n_epochs=500,n_batchsize=64)
        plot_balanced_accuracy(model_history)

    ###getting the validation scores for the different models
    val_score_logistic=get_val_score(X_data, labels,logistic_model)
    val_score_svm=get_val_score(X_data, labels,svm_model)
    
    print(val_score_logistic,val_score_svm)

    ###training the model to predict
    logistic_model1 = train_logistic_model(X_data, labels) #training the model
    svm_model1=train_svm_model(X_data, labels)

    ###preparing the test data
    test_data=reading_data('test.csv') #get the test_data out of the excel file
    processed_test_data=preparing_test_data(test_data, clean_data, pca_model, manner, fingerprintcount)
    #processed_test_data_svm=preparing_test_data(test_data, clean_data_svm, pca_model_svm, manner, fingerprintcount)

    ###predicting the test set
    output_predictions_logistic = predict_test_data(processed_test_data, logistic_model1)
    output_predictions_svm = predict_test_data(processed_test_data, svm_model1)
    # print(output_predictions.head)

    ###Save predictions to CSV
    if write_to_file == True:
        print("save predictions to CSV")
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        output_csv_path = os.path.join(script_dir, 'predicted_outcomes_logistic_aug5_fp4.csv')
        output_predictions_logistic.to_csv(output_csv_path, index=False)

        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        output_csv_path = os.path.join(script_dir, 'predicted_outcomes_svm_aug5_fp4.csv')
        output_predictions_svm.to_csv(output_csv_path, index=False)

    print('this is the validation score for logistic and svm', val_score_logistic, val_score_svm)
    if manner == 'completely':
        print('this is the best cor and var for logistic', best_corr_logistic, best_var_logistic)
        print('this is the best cor and var for svm', best_corr_svm,best_corr_logistic)
    descriptors_time=descriptors_end_time-descriptors_start_time
    cleaning_time=cleaning_end_time-cleaning_start_time
    pca_time=pca_end_time-pca_start_time

    ###when running this code with fingerprint bit count 1024 and 5 times augmentation you get a validation score for logistic regression of 0.9984844408427878
    print('this is the time it takes to get the descriptors', descriptors_time)
    print('this is the time it takes to clean the data', cleaning_time)
    print('this is the time it takes to perform pca', pca_time)
