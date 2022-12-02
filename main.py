import streamlit as st
import util as utl
# from views import home,about,analysis,options,configuration
from views import home, analysis, trynow

st.set_page_config(layout="wide", page_title='Navbar sample')
st.set_option('deprecation.showPyplotGlobalUse', False)
utl.inject_custom_css()
utl.navbar_component()

def navigation():
    route = utl.get_current_route()
    if route == "home":
        home.load_view()
    elif route == "analysis":
        analysis.load_view()
    elif route == "trynow":
        trynow.load_view()
    
    elif route == "configuration":
        # configuration.load_view()
        None
    elif route == None:
        home.load_view()
        
navigation()