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
from sklearn import metrics





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
# print(data)
def return_classifier():


    st.title('Sequence your data for classification')
    option = st.selectbox(
        'Which dataset do you want to view?',
        (i for i in data_files), key=1)
    dataset = pd.read_csv(option)


    def slice_data(time_series,seq_len):
        time_series = np.array(time_series)
        data = np.zeros((0,seq_len,time_series.shape[1]))

        idx_last = -(time_series.shape[0] % seq_len)
        if idx_last < 0:    
            clips = time_series[:idx_last].reshape(-1, seq_len,time_series.shape[1])
        else:
            clips = time_series[idx_last:].reshape(-1, seq_len,time_series.shape[1])                

        # Partition train and test set in separate arrays
        #print(n)
        data = np.vstack((data, clips))
        print(data.shape)

        return(data)

    length = st.slider(label='choose length',min_value=5, max_value=len(dataset), value=10, step=5, key=2)

    if length > 2:
    
        data2 = slice_data(dataset, seq_len=length)
        print('a calculation is done')
        st.write(f"the shape your object is {data2.shape}")



        
        

