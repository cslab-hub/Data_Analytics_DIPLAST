import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from PIL import Image 

from home import *
from matrix_profile import *
from comparison import *
from preprocessing import *
from forecast import *
from sequencing_classifier import *
# from columnnames import *
# from texteditors import *
# from number_of_variables import *

def select_block_container_style():
    max_width_100_percent = st.sidebar.checkbox("Max-width: 100%?", False)
    if not max_width_100_percent:
        max_width = st.sidebar.slider("Select max-width in px", 100, 2000, 1200, 100)
    else:
        max_width = 1200

    _set_block_container_style(
        max_width,
        max_width_100_percent,

    )


def _set_block_container_style(
    max_width: int = 1200,
    max_width_100_percent: bool = False,
):
    if max_width_100_percent:
        max_width_str = f"max-width: 100%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    }}
</style>
""",
        unsafe_allow_html=True,
    )
    
    
st.sidebar.title("Select Tool")
st.sidebar.header("Each tool performs a different task.")




add_selectbox = st.sidebar.radio(
    "Choose a chapter:",
    ("Home",'Preprocessing',"Matrix Profile",'Comparison',"Forecast","Classifier"),format_func= lambda x: 'Home' if x == 'Home' else f"{x}"
    
)                                                                                                                       


#! Home page
if add_selectbox == 'Home':
    return_homepage()
    
    
#! Page for the file format
if add_selectbox == 'Matrix Profile':
    return_matrix_profile()

if add_selectbox == 'Comparison':
    return_comparison()
    
if add_selectbox == 'Preprocessing':
    return_preprocessing()

if add_selectbox == 'Forecast':
    return_forecast()

if add_selectbox == 'Classifier':
    return_classifier()
    
select_block_container_style()
# This removes the copyright of how the page is made
hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)