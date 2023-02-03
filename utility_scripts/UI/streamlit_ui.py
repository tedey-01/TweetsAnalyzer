import requests
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio

from plot_tools import prepare_df_for_plot_on_map

pio.templates.default = "seaborn"


AI_URL = "http://127.0.0.1:5000/analyze_tweet/"
PLOTTING_DATA_PATH = os.path.join('..', '..', 'data', 'df_for_plot_on_map.csv')


class Tweet:
    def __init__(self, text: str, keyword: str, location: str):
        self.text = text
        self.keyword = keyword
        self.location = location
        self.size = 40
        self.id = -1
        self.target = None

    def convert_to_df(self) -> pd.DataFrame:
        tweet_df = pd.DataFrame({
            'id': self.id,
            'keyword': self.keyword,
            'location': self.location,
            'text': self.text,
            'target': self.target,
            'size': self.size,
        }, index=[0])
        return tweet_df


def analyze_tweet(url: str, tweet: Tweet) -> dict:
    data = {
        'text': [tweet.text],
        'keyword': [tweet.keyword],
        'location': [tweet.location]
    }
    resp = requests.post(url, json={**data})
    resp = json.loads(resp.text)
    return resp


def concat_plot_data(data_path: str, tweet: Tweet) -> pd.DataFrame:
    tweet_df = tweet.convert_to_df()
    tweet_df = prepare_df_for_plot_on_map(tweet_df)
    df = pd.read_csv(data_path)
    df = pd.concat([df, tweet_df])
    return df


def create_barplot(probabilities: list):
    fig = px.bar(
        x=['real', 'fake'],
        y=probabilities,
        title="Probabilities of belonging to a class",
        labels=dict(x="Classes", y="Probability")
    )
    fig.update_traces(
        marker_color='rgb(255,255,102)',
        marker_line_color='rgb(255,153,0)',
        marker_line_width=1.5,
        opacity=0.6,
    )
    # fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)
    return fig


def create_scatterplot(df: pd.DataFrame):
    map_fig = px.scatter_geo(
        df,
        lat='lat',
        lon='lon',
        color='target',
        hover_data=['text'],
        opacity=0.7,
        size='size',
        projection="natural earth",
    )
    return map_fig


# Общий функционал и настройки
st.set_page_config(layout='wide')
st.sidebar.markdown('### Настройки')
doc_types = ['Log Reg', 'BERT']
selected_doc_type = st.sidebar.selectbox('Модель', doc_types)

col1, _, col2 = st.columns([10, 2, 14])
with col1:
    # Выбор файла с признаками обязателен для дальнейшего выполнения.
    st.markdown("### :red[1. Add Tweet]", )
    tweet_body = st.text_area("Write Tweet Body", placeholder="There is a fiery sunset in Moscow today")
    tweet_keywords = st.text_input("Write keywords separated by a space", placeholder="#nature")
    tweet_location = st.text_input("Write your location", placeholder="Moscow")

    process_button = st.button('Обработать')
    if process_button:
        tweet = Tweet(tweet_body, tweet_keywords, tweet_location)
        resp = analyze_tweet(AI_URL, tweet)
        tweet.target = resp['class']
        st.json(resp)
with col2:
    if process_button:
        # Show Map
        st.markdown("### :red[2. Plot on MAP]")
        data_plot = concat_plot_data(PLOTTING_DATA_PATH, tweet)
        map_fig = create_scatterplot(data_plot)
        st.plotly_chart(map_fig, use_container_width=True)

        # Show barplot
        bar_fig = create_barplot(resp['probabilities'])
        st.plotly_chart(bar_fig, use_container_width=True)


st.stop()
