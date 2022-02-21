import streamlit as st
# st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from PIL import Image 

import os 
import stumpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import scipy as sp
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn import metrics
import seaborn as sns

classifier_list = ["RandomForestClassifier","KNeighborsClassifier"]


def data_loader():
    found_files = []
    cwd = os.getcwd()
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                print(filename)
                # data = pd.read_csv(os.path.join(roots,filename))
                found_files.append(os.path.join(roots,filename))
    return found_files

data_files = data_loader()
data_files.insert(0,'Select a Dataset')
# print(data)
def return_classifier():


    st.title('Choose your data for classification')

    st.markdown("""
        One of the most used applications in the field of Data Science is Classification. The idea behind classification is to predict the class, also called target variable, of the given data points. 
        For example, spam detection by your e-mail client can be seen as a classification problem. In this case, we talk about a binary classification problem since there are only two outcomes,
        e.g., Spam (0) and Not Spam (1). A classifier, which is an algorithm used to learn from the data how the input variables relate to the specific class. Generally speaking,
        the data will be divided into a train and test set. In the e-mail use case, known spam and non-spam e-mails are used as training data. If the classifier is trained accuractely, 
        it can be used to detect unknown e-mails. The test data, which is separated in an earlier stage, is used to validate the performance of a given classifier. If the performance is good enough on 
        unseen data (the test data), we can assume that with future instances, the classifier will be able to make the distinction between classes. 

        """)

    option = st.selectbox(
        'Which dataset do you want to use for your classification problem?',
        (i for i in data_files),format_func= lambda x:  str(x).split('/')[-1], key=1)

    if option == "Select a Dataset":
        st.stop()
    dataset = pd.read_csv(option)

    st.table( dataset.head(3))




    option2 = st.selectbox(
        'Which variable resprents the labels for the given dataset? We will separate this variable from the rest of the dataset ', 
        (i for i in dataset.columns), key=1)
    target_data = dataset[option2]
    dataset = dataset.drop(columns = [option2])


    st.markdown(f""" Now that we have identified which data we will use to classify the labels represented by {option2}, we can start to proceed with making a classifier.
        Please note in the table below that the column containing the different classes is removed.""") 
    st.table( dataset.head(3))





    st.markdown("""
    


        The next step is to determine the sequence length that you expect in your data. This can be adapted by using the slider below. If you keep the slider at zero, the data will not be separated in sequences and will be fed to the classifier point-by-point.


        """)



    # def slice_data(time_series,seq_len):
    #     time_series = np.array(time_series)
    #     data = np.zeros((0,seq_len,time_series.shape[1]))

    #     idx_last = -(time_series.shape[0] % seq_len)
    #     if idx_last < 0:    
    #         clips = time_series[:idx_last].reshape(-1, seq_len,time_series.shape[1])
    #     else:
    #         clips = time_series[idx_last:].reshape(-1, seq_len,time_series.shape[1])                

    #     # Partition train and test set in separate arrays
    #     #print(n)
    #     data = np.vstack((data, clips))
    #     print(data.shape)

    #     return(data)


    # length = st.slider(label='choose length',min_value=0, max_value=len(dataset), value=0, step=5, key=2)

    
    # if length < 2:
    #     data2 = np.array(dataset)
    #     st.write(f"the shape your object is {data2.shape}, which will be used to train our classifier.")
    # else:
    #     data2 = slice_data(dataset, seq_len=length)
    #     print('a calculation is done')
    #     st.write(f"the shape your object is {data2.shape}, which will be used to train our classifier.")



    option3 = st.selectbox(
        'Which classifier do you want to use?',
        ('Select algorithm','K Nearest Neighbors', 'Random Forest'), key= 1)
        # (i for i in classifier_list), key=1)
    x_train, x_test, y_train, y_test= train_test_split(dataset, target_data,
                                                   test_size= 0.2,
                                                   shuffle= True, #shuffle the data to avoid bias
                                                   random_state= 0)


    scaler= StandardScaler()
    scaled_x_train = scaler.fit_transform(x_train)
    scaled_x_test = scaler.transform(x_test)
    st.write(f'training set size: {x_train.shape[0]} samples \ntest set size: {x_test.shape[0]} samples')

    
    if option3 == "K Nearest Neighbors":
        k_range = list(range(1,50))
        weight_options = ["uniform", "distance"]
        param_grid = dict(n_neighbors = k_range, weights = weight_options)
        classifier = KNeighborsClassifier()


    if option3 == "Random Forest":

        classifier = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

        param_grid = { 
            'n_estimators': range(1,120,10),
            'max_features': ['auto', 'sqrt', 'log2']
            }


    cv_classifier = GridSearchCV(estimator=classifier, param_grid=param_grid, cv= 5, verbose = 100, n_jobs = -1)
    cv_classifier.fit(scaled_x_train, y_train)
    feature_importance = cv_classifier.best_estimator_.feature_importances_
    f_i = list(zip(features_dataset,feature_importance)) 
    f_i.sort(key = lambda x : x[1])
    fig2, ax2 = plt.subplots()
    ax2 =  plt.barh([x[0] for x in f_i],[x[1] for x in f_i]) 
    ax2.set_yticklabels(fontsize=16)
    st.pyplot(fig2)
    
    
    sorted(cv_classifier.cv_results_.keys())
    cv_best = cv_classifier.best_estimator_
    predictions = cv_best.predict(scaled_x_test)



    mat = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots()
    ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d')
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label');

    classifier_accuracy = round(metrics.accuracy_score(y_test, predictions),4)

    st.write(f"Based on the analysis using the {option3} algorithm, the classifier was able to predict the right class with an accuracy of {classifier_accuracy} and parameter settings {cv_best}")
    st.pyplot(fig)



    
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))

    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, scaled_x_train, y_train, cv=kfold, scoring='accuracy', n_jobs = -1, verbose = 2)
        results.append(cv_results)
        names.append(name)
        st.write('%s: %f (%f)' % (name, round(cv_results.mean(),2), cv_results.std()))


        
    print(np.unique(y_test, return_counts =True))
    print(np.unique(y_train, return_counts =True))



        # sns.heatmap


# # Print classification report and create confusion matrix
# print(classification_report(y_test, predictions))

    





        
        

