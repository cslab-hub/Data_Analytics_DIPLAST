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
data.insert(0,'Select a Dataset')

def return_preprocessing():


    st.title('Preprocess your data')
    option = st.selectbox(
            'Which dataset do you want to view?',
            (i for i in data), format_func= lambda x:  str(x).split('/')[-1], key=1)
    if option == "Select a Dataset":
        st.stop()
    
    dataset = pd.read_csv(option)


    
    for col in dataset.columns:
        dataset.loc[dataset.sample(frac=0.1, random_state=3).index, col] = np.nan
    st.write("""
        The dataset below shows the first 10 inputs. Based on this information, you are able to see the general outline of the dataset, e.g., the amount of columns and some values. 
        """)
    st.table(dataset.head(10))
    st.write(f'The dataset contains a total of {len(dataset)} rows and {len(dataset.columns)} columns.')
    dtype_df = dataset.dtypes.value_counts().reset_index()

    dtype_df.columns = ['VariableType','Count']
    dtype_df['VariableType'] = dtype_df['VariableType'].astype(str)
    # dtype_fig = plt.bar(dtype_df['VariableType'],dtype_df['Count'])
    # print(dtype_df)
    # p1 = plt.bar(pos, 1, color='red', data=data)

    fig, ax = plt.subplots()

    # fig.set_size_inches(24.5, 16.5)
    ax.bar(dtype_df['VariableType'],dtype_df['Count'])
    
    st.write("""
        The plot below indicates which datatypes are in the dataset, based on the different columns (variables). The barplot shows the different datatypes and how many of these columns are spotted in the dataset.
        """)

    st.pyplot(fig)

    st.write("""
        As can be seen in the table above, the dataset contains some missing values "<NA>". For analysis purposes, it would be good to handle these missing values.\n
        There are two general ways to do so; delete the entire row where the missing value occurred, or impute the average value of the column op the place of the missing value.\n
        Both options have their pros and cons, and it is up to you to decide which method is well-suited for your dataset. Below you will be able to make this decision. 
        """)

    option2 = st.selectbox(
            'Which action do you want to take?',
            ('Select action','Delete','Infer'), key=2)
    
    if option2 == 'Delete':
        st.write('Result when deleting the rows with faulthy data')
        
        st.table(dataset.dropna().head(10))
        
    if option2 == 'Infer':
        st.write('Result when inferring the values from faulthy data')
        
        # st.dataframe(dataset.dropna())
        st.table( dataset.fillna(dataset.rolling(4,min_periods=1).mean()).head(10) )

    
