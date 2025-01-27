import base64
import importlib
import pandas as pd
import streamlit as st

def render_page(menupage):
    menupage.write()

def title_awesome(body: str):
    st.title(
        "Barbenheimer Data Analytics:"  
        f"{body} "
        "[![Quantum](https://cdn.rawgit.com/sindresorhus/awesome/"
        "d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)]"
    )

def render_md(md_file_name):
    st.markdown(get_file_content_as_string(md_file_name))

def get_file_content_as_string(path):
    response = open(path, encoding="utf-8").read()
    return response

def show_code(file_name):
    return get_file_content_as_string(file_name)

def data_frame(filename):
    df = pd.read_csv(filename)
    st.write(df)