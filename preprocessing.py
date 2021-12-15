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

# dataframe = pd.DataFrame(np.random.randint(0,100,size=(100, 4)))
# dataframe.columns = ['var1','var2','var3','var4']
# dataframe.to_csv('data/dataset2.csv', index=False)



def data_loader():
    found_files = []
    cwd = os.getcwd()
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                found_files.append(os.path.join(roots,filename))
    return found_files

data = data_loader()

def return_preprocessing():

    col1, col2, col3 = st.columns(3)

    with col2:
        option = st.selectbox(
                'Which dataset do you want to view?',
                (i for i in data), key=1)
        dataset = pd.read_csv(option)
        
        for col in dataset.columns:
            dataset.loc[dataset.sample(frac=0.1, random_state=3).index, col] = np.nan
        st.table(dataset.head(10))


        option2 = st.selectbox(
                'Which action do you want to take?',
                ('Delete','Infer'), key=2)
        
        if option2 == 'Delete':
            st.write('Result when deleting the rows with faulthy data')
            
            st.table(dataset.dropna().head(10))
            
        if option2 == 'Infer':
            st.write('Result when inferring the values from faulthy data')
            
            # st.dataframe(dataset.dropna())
            st.table( dataset.fillna(dataset.rolling(4,min_periods=1).mean()).head(10) )