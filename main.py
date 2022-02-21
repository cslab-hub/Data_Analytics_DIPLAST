import streamlit as st
st.set_page_config(page_title="Di-Plast Data Analytics Tool ",layout="wide")
# st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from PIL import Image 

from home import *
from matrix_profile import *
from comparison import *
from preprocessing import *
# from forecast import *
from ts_classifier import *
from final_report import *
from feature_selection import *
import base64
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

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_img_with_href(local_img_path, target_url,width="1"):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}" target="_blank">
            <img style="width: {width}%" src="data:image/{img_format};base64,{bin_str} " />
        </a>'''
    return html_code



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
    
Logo_html = get_img_with_href('di-plast_Logo.PNG', 'https://www.nweurope.eu/projects/project-search/di-plast-digital-circular-economy-for-the-plastics-industry/',width="90")    
st.sidebar.markdown(Logo_html, unsafe_allow_html=True)
st.sidebar.title("Select a Module")
# st.sidebar.header("Each tool performs a different task.")





add_selectbox = st.sidebar.radio(
    "Choose one of the analytics options:",
    ("Home",'Preprocessing',"Feature Selection","Matrix Profile",'Comparison',"Classifier",'Final Data Report'),format_func= lambda x: 'Home' if x == 'Home' else f"{x}",help="Please select one of the options that aligns with your analytics needs."
    
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

if add_selectbox == 'Feature Selection':
    return_feature_selection()

# if add_selectbox == 'Forecast':
#     return_forecast()

if add_selectbox == 'Classifier':
    return_classifier()

if add_selectbox == 'Final Data Report':
    return_report()
    
select_block_container_style()
# This removes the copyright of how the page is made
hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)



JADS_logo = get_img_with_href('JADS_logo.png', 'https://www.jads.nl',width="90")
st.sidebar.markdown(JADS_logo, unsafe_allow_html=True)
st.sidebar.caption("[Bug reports and suggestions welcome ](mailto:j.o.d.hoogen@jads.nl)")