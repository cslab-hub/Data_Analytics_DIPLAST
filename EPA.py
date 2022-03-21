# %%
import numpy as np
import pandas as pd
import re
import streamlit as st
from PIL import Image 
import os 
import matplotlib.pyplot as plt 
import datetime as dt
import sys 
import csv
import sd4py
import sd4py_extra
import jpype

# %%

def data_loader():
    found_files = []
    cwd = os.getcwd()    
    cwd = cwd+'/data/preprocessed'
    print(cwd)
    for roots, dirs, files in sorted(os.walk(cwd)):
        for filename in sorted(files):
            if filename.endswith(".csv"):
                found_files.append(os.path.join(roots,filename))
    return found_files

data = data_loader()
data.insert(0,'Select a Dataset')

# %%
def return_EPA():




    st.title('Descriptive statistics of your data')
    option = st.selectbox(
        'Which dataset do you want to view?',
        (i for i in data), format_func= lambda x:  str(x).split('/')[-1], key=1)
    if option == 'Select a Dataset':
        st.stop()
    dataset = pd.read_csv(option,index_col=0)

    st.write("""
        The dataset below shows the first 10 inputs. Based on this information, you are able to see the general outline of the dataset, e.g., the amount of columns and some values. 
        """)

    st.dataframe(dataset.head(10))
    st.write(f'The dataset contains a total of {len(dataset)} rows and {len(dataset.columns)} columns.')


    dataset.index = pd.to_datetime(dataset.index)


    state_changes = dataset.State[dataset.State.values != dataset.State.shift(1)]
    dataset_state_changes = dataset.join(pd.Series(state_changes.index, state_changes.index, name='change_time'), how='left').backfill()
    dataset_state_changes = dataset_state_changes.join(state_changes, how='left', rsuffix='_new').backfill()
    dataset_production = dataset_state_changes[dataset_state_changes.State == 1]

    dataset_production.insert(dataset_production.shape[1], 
                             'within_30', 
                             (dataset_production.index.values - dataset_production.change_time.values) > pd.Timedelta(-30, unit='minutes'))

    ## Get rid of that pesky first column
    dataset_production = dataset_production.iloc[1:]
    option2 = st.selectbox(
        'Do you want to explore patterns in your dataset?',
        ['No','Yes'], key=1)
    if option2 == 'No':
        st.stop()
    # %%
    subgroups = sd4py.discover_subgroups(dataset_production.drop(columns=['State', 'change_time', 'State_new']), "within_30", qf='bin',
                                    k=100, postfilter="relevancy")

    # %%
    subgroups.to_df()

# %%
#pd.set_option("display.max_colwidth", 100)

# %%
    import warnings

    with warnings.catch_warnings():
        
        warnings.simplefilter("ignore")

        with_confidence = subgroups.to_df().merge(
            sd4py_extra.confidence_precision_recall_f1(subgroups, 
                                                       dataset_production.drop(columns=['State', 'change_time', 'State_new']), 
                                                       "within_30", 
                                                       value=True,
                                                       number_simulations=100
                                                      )[1], 
            on="pattern")

    # %%
    subgroups_selection = [subgroups.subgroups[i] for i in with_confidence[with_confidence.f1_lower > 0.03].index]

    # %%
    with_confidence[with_confidence.f1_lower > 0.03]

    # %%
    with warnings.catch_warnings():
        
        warnings.simplefilter("ignore")
        
        results_dict, aggregation_dict = sd4py_extra.confidence_intervals(subgroups_selection, 
                                                       dataset_production.drop(columns=['State', 'change_time', 'State_new']), 
                                                       "within_30", 
                                                       value=True)

    ## To make the subgroup names more readable
    fig, ax = plt.subplots(figsize=(7,3))
    ax = sd4py_extra.confidence_intervals_to_boxplots(results_dict, labels=[re.sub('AND', '\nAND',key) for key in results_dict.keys()])
    ax = plt.yticks(fontsize=12)
    ax = plt.xticks(fontsize=12)
    ax = plt.xlabel('Proportion of Subgroup Members that Had Fault within 30 Minutes', size=12)
    ax = plt.gca().set_title('Distribution of Mean Target Value from Bootstrapping',pad=20)
    ax = plt.gcf().set_size_inches(17,10)
    ax = plt.gcf().tight_layout()

    st.pyplot(fig)
    # st.pyplot(fig.figure)



## CHeck the rest afterwards
    # %%
    fig, ax = plt.subplots(figsize=(7,3))
    ax = sd4py_extra.jaccard_visualisation(subgroups_selection, 
                                           dataset_production.drop(columns=['State', 'change_time', 'State_new']), 
                                           1/5, 
                                           labels=[re.sub('AND', '\nAND',key) for key in results_dict.keys()])

    ax = plt.gcf().set_size_inches(20,9)
    ax = plt.margins(x=0.15)
    ax = plt.gca().set_frame_on(False)
    ax = plt.gca().set_title('Jaccard Similarity Between Subgroups', fontsize=14)
    ax = plt.gcf().tight_layout()
    st.pyplot(fig)

# # %%


    # %%
    plt.figure()

    saved_figsize = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (24,24)

    plt.gcf().suptitle(re.sub('AND', '\nAND',str(subgroups_selection[0])), y=0.95)

    plt.tight_layout()

    sd4py_extra.subgroup_overview(subgroups_selection[0], 'within_30', dataset_production.drop(columns=['State', 'change_time', 'State_new']), axis_padding=50)

    st.pyplot(plt.gcf())

    plt.rcParams["figure.figsize"] = saved_figsize

# # %%



