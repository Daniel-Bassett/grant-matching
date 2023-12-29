import time
import io

import pandas as pd
import numpy as np

import streamlit as st
import plotly_express as px
from streamlit_option_menu import option_menu

from openai import OpenAI

try:
    from config import *
except:
    pass

try:
    client = OpenAI(api_key=API_KEY)
except:
    client = OpenAI(api_key=st.secrets["api_key"])


st.set_page_config(layout='wide')

@st.cache_data
def load_data(path):
    if '.parquet' in path:
        return pd.read_parquet(path)
    if '.csv' in path:
        return pd.read_csv(path)


@st.cache_data
def concat_df(list_of_df):
    grants = pd.concat(list_of_df).reset_index(drop=True)
    return grants


def get_embedding(text, model="text-embedding-ada-002"):
    return client.embeddings.create(input = [text], model=model).data[0].embedding


def clear_query():
    st.session_state['query_text'] = ""
    st.session_state['button1'] = False


def convert_df(df):
    # Create a BytesIO buffer
    output = io.BytesIO()
    
    # Use ExcelWriter with the buffer as the file
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
        writer.close()

    # Rewind the buffer
    output.seek(0)

    # Return the buffer's content in a format suitable for streamlit.download_button
    return output.getvalue()


def process_df(df):
    df = df.rename(columns={'parent_name': 'agency'})
    df = df.rename(columns={'sbir_topic_link': 'url'})
    df = df.rename(columns={'topic_title': 'title'})
    df = df.rename(columns={'opportunity_number': 'topic_number'})
    df = df.query('text.str.len() > 250')
    df = df.reset_index(drop=True)
    return df

# load data
dod = load_data('data/dod_processed.parquet')
doe = load_data('data/doe_processed.parquet')
hhs = load_data('data/hhs_processed.parquet')

dod = process_df(dod)
doe = process_df(doe)
hhs = process_df(hhs)

grants = concat_df([dod, doe, hhs])


if "button1" not in st.session_state:
    st.session_state["button1"] = False

if "button2" not in st.session_state:
    st.session_state["button2"] = False




# Text Output
query = st.text_area('Enter description:', max_chars=3000, key='query_text', height=300)

col1, col2, col3, col4 = st.columns([3, 3, 5, 3])
# col1, col2, col3 = st.columns([2, 5, 9])

with col1:
    find_program = st.button('Find Programs')
with col2:
    delete_abstract = st.button('Clear Keywords', on_click=clear_query)

# filter_columns = st.columns([5, 7])
# with filter_columns[0]:
#     with st.expander('Filters'):
#         agencies = st.multiselect(label='Choose Agency', options=grants['agency'].unique())

if find_program:
    st.session_state['button1'] = True

if st.session_state['button1']:
    query_embedding = get_embedding(query)
    similarity_index = grants['text_embedded'].apply(lambda y: np.dot(query_embedding, y)).sort_values(ascending=False).head(30)
    similarity_index = similarity_index.index
    st.data_editor(grants.iloc[similarity_index][['title', 'text', 'topic_number', 'url']],
                hide_index=True,
                use_container_width=True,
                )


    # for index, row in grants.iloc[similarity_index].iterrows():
    #     st.markdown(f'#### {row["title"]}')
    #     with st.expander('Show Text'):
    #         st.write(row['text'])
    #     st.write(row['url'])


        # st.write('Number:', row['number'])
        # if st.session_state['button1']:
        #     if st.button('AI Report', key=f'ai{index}'):
        #         text = row['text']
        #         completion = client.chat.completions.create(
        #         model="gpt-3.5-turbo-1106",
        #         messages=[
        #             {"role": "system", "content": "You aid people in understand grant proposals. Your purpose is to compare the abstract of a proposal to the grant program description. Explain in bulleted format in less than 700 words why the abstract and grant program match or don't match. Be sure to mention specific technologies, innovations, research topics, etc. that are in the provided grant and abstract text."},
        #             {"role": "user", "content": f"Here is my abstract: {query}. Here is the grant description: {text}"}
        #         ]
        #         )
        #         st.text_area(label='', value=f'{completion.choices[0].message.content}', height=700)

        # st.divider()

