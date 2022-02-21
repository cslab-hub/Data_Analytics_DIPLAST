## Feature selection
from select import select
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt 
import os

# import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests

#%%

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

data = data_loader()
data.insert(0,'Select a Dataset')

def return_feature_selection():
    st.title('Create Correlation plots')


    # st.markdown("""Correlation is a statistical term which refers to how close two variables have a linear relationship to each other.
    # Variables that have a linear relationship tell us less about our dataset, since measuring one tells you something about the other.
    # In other words, if two variables have a high correlation, we can drop on of the two!
    # """)

    option = st.selectbox(
    'Which dataset do you want to view?',
    # ['Select dataset',(i for i in data)], format_func= lambda x:  str(x).split('/')[-1], key=1)
    (i for i in data), format_func= lambda x:  str(x).split('/')[-1], key=1)

    if option == "Select a Dataset":
        st.stop()

    dataset = pd.read_csv(option)


    st.table(dataset.head(5).style.format(precision=2)\
    .set_table_styles([
                    {"selector":"caption",
                    "props":[("text-align","center")],
                    }

                    ], overwrite=False)\

        .set_caption('Table 1.'))
    corr = dataset.corr().round(2)
    corr.style.background_gradient(cmap='coolwarm')
    st.table(corr.style.background_gradient(cmap='coolwarm')\
    .format(precision=2)\
    .set_table_styles([
                    {"selector":"caption",
                    "props":[("text-align","center")],
                    }

                    ], overwrite=False)\
        

        .set_caption('Table 2.'))



    # build the time series, just a simple AR(1)]



    #######################################################
    st.title('PCA Analysis')
    st.markdown('''
    A technique to reduce the dimensionality of your dataset is by performing Principal Component Analysis.
    PCA uses a set of large variables by combining them together to retain as much as information as possible.
    PCA dates back to the 1990's and is one of the most widely used analysis techniques in Data Science.
    ''')

    from sklearn.preprocessing import StandardScaler # for standardizing the Data
    from sklearn.decomposition import PCA # for PCA calculation

    option2 = st.selectbox(
        'Which variable should be removed from the dataset?',
        (i for i in dataset.columns), key=2)
    # fig = plt.plot(plot[option2])

    variables = dataset[option2]

    # df = pd.read_csv('data/Turbine_Data.csv', parse_dates=["Unnamed: 0"])
    # dataset['DateTime'] = df['Unnamed: 0'] 
    # dataset.drop('Unnamed: 0', axis=1, inplace=True)
    # Add datetime parameters 
    # dataset['DateTime'] = pd.to_datetime(df['DateTime'], 
    # format = '%Y-%m-%dT%H:%M:%SZ', 
    # errors = 'coerce')



    # Drop Blade1PitchAngle columns due to high number of missing values
    # df = df.drop(['Blade1PitchAngle', 'Blade2PitchAngle', 'Blade3PitchAngle'], 1)
    # df.fillna(df.mean(), inplace=True)
    # Drop WTG column because it doesn't add much
    # df.drop('WTG', axis=1, inplace=True)

    # # Drop ControlBoxTemperature column because it doesn't add much
    # df.drop('ControlBoxTemperature', axis=1, inplace=True)

    # # st.dataframe(df.head(40))
    # all_values = df.copy()
    # # all_values.to_csv('data/delimiter_tests/turbine.csv', index=False)
    # features = ['WindSpeed', 'RotorRPM', 'ReactivePower', 'GeneratorWinding1Temperature', 
    # 'GeneratorWinding2Temperature', 'GeneratorRPM', 'GearboxBearingTemperature', 'GearboxOilTemperature']

    # # Separate data into Y and X 
    # y = all_values['ActivePower']
    X = dataset

    # st.dataframe(X.head(40))
    # print(y.shape)
    print(X.shape)

    sc = StandardScaler() # creating a StandardScaler object
    X_std = sc.fit_transform(X) # standardizing the data

    pca = PCA()
    X_pca = pca.fit(X_std)

    def pcaplotter():
        fig, ax = plt.subplots(figsize=(8,3))
        # max_bins = int(max(ax.get_xlim()) + 2)
        # print('max bins = ',max_bins)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        # ax.set_xticks([i for i in range(1,max(ax.get_xlim())+2)])
        # ax.set_xticks([i for i in range(1,max_bins,1)])
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().locator_params(nbins=max(ax.get_xlim())+1)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')

        return fig

    # st.pyplot(pcaplotter())

    # num_components = 4
    # pca = PCA(num_components)  
    # X_pca = pca.fit_transform(X_std) # fit and reduce dimension
    # print(pca.n_components_)
    aim_target = st.slider('How much variance should be explained?', min_value=0.9, max_value=0.99, step=0.01, value=0.95)
    pca = PCA(n_components = aim_target)
    X_pca = pca.fit_transform(X_std) # this will fit and reduce dimensions
    # st.markdown(f'{pca.n_components_}') # one can print and see how many components are selected. In this case it is 4 same as above we saw in step 5

    st.pyplot(pcaplotter())
    pd.DataFrame(pca.components_, columns = X.columns)

    n_pcs= pca.n_components_ # get number of component
    # get the index of the most important feature on EACH component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = X.columns
    # get the most important feature names
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    # st.markdown(f'The most outstanding variables in your dataset are in order from important to less important: {most_important_names}')

    for i,j in enumerate(most_important_names):
        st.write(f"{i + 1}th most important variable = {j}")

