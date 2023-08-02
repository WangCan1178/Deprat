import streamlit as st
import numpy as np
import pandas as pd

import streamlit as st

st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# tab1, tab2 = st.columns(2)
# with tab1:
st.write("# Welcome to API parameter descriptions automatic generation platform! 💕")

# st.sidebar.success("Select a demo above.")
st.markdown(
    """
    This Web is an open-source web built by streamlit for viewing, editing, and managing your services. 
    Use natural language generation technology to help you quickly generate parameter descriptions and API documents. ^_^
    
    **👈 Switching tabs from the sidebar** to see  of what Streamlit can do!
    ### Want can you do?
    - View, edit, and manage your services, which in the "[Services](https://w-caner-clothingrecapp-welcome-2g8n9r.streamlit.app/Clothing_shop)" tab
    - Automatically generate parameter descriptions and download API documentation, and this function also in the "[Services](https://w-caner-clothingrecapp-welcome-2g8n9r.streamlit.app/Clothing_shop)" tab
    - Evaluate the quality of the parameter description, using a metric we defined in paper, this function in the "[Evaluate](https://w-caner-clothingrecapp-welcome-2g8n9r.streamlit.app/My_Cart)" tab

    ### See more information!
    - Methods and experiments about thr paper provided to show details, visit by clicking the "[Information](https://w-caner-clothingrecapp-welcome-2g8n9r.streamlit.app/Show_Analysis)" tab
    - The project is open source on github at [https://github.com/W-caner/ClothingRecApp.git](https://github.com/W-caner/ClothingRecApp.git)
    
    # [GETTING START NOW!](https://w-caner-clothingrecapp-welcome-2g8n9r.streamlit.app/Deprat)
"""
)
