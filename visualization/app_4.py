#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats 
from matplotlib.ticker import MaxNLocator

import streamlit as st
from PIL import Image # Required to show images
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# In[41]:


# just for production ... should be done in seperate Notebook
# In[ ]:
# Page settings
st.set_page_config(
	layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="collapsed",  # Can be "auto", "expanded", "collapsed"
	page_title=None,  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
)

import base64

main_bg = "background.png"
main_bg_ext = "png"

#side_bg = "sample.jpg"
#side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }} </style>""", unsafe_allow_html=True)


audio_file = open("./MUSIC/background_music.mp3", "rb")
audio_bytes = audio_file.read()
st.audio(audio_bytes, format="audio/ogg")

st.markdown("# Ready for the adventure ?")
passw=st.radio("Enter the password",("Dont know","Alohomora","wingardium leviosa"))
if passw == "Alohomora":
    st.title("Welcome to the magical world of fantasy.")
    st.markdown("![Welcome to Hogwarts](https://media.giphy.com/media/3oFzmbQom9DGp7OhZS/giphy.gif)")
    st.header('We are team Hufflepuff')
    st.markdown("![The world of fantasy](https://media.giphy.com/media/PMp40oEvNfKve/giphy.gif)")
    
    # TITLE
        
    # Show our data

    st.header("Explore the data of the 1000 best epic fantasy books")

    st.markdown("""Epic fantasy is generally serious in tone and often epic in scope, dealing with themes of grand struggle against supernatural, evil forces. Some typical characteristics of epic fantasy include fantastical elements such as elves, fairies, dwarves, magic or sorcery, wizards or magicians, invented languages, quests, coming-of-age themes, and multi-volume narratives.""")

    df = pd.read_csv("Data_graphs.csv")
    if st.button("Show me the data."):
        st.dataframe(df)

    # Choose columns to show
    st.markdown("If you want to display only few columns, you can chose here:")
    columns_to_show = st.multiselect('', df.columns)

    # Books with rating higher than X
    st.markdown("Just show books with a rating higher than:")
    s1_min = float(df.minmax_norm_ratings.min())
    s1_max = float(df.minmax_norm_ratings.max())
    threshold = st.slider("", s1_min, s1_max)
    filtered = df[df["minmax_norm_ratings"] >= threshold]
    st.dataframe(filtered[columns_to_show])

    # Sidebar
    st.sidebar.header("Who is your favourite fantasy author?")
    #col1 = st.sidebar.beta_columns(1)
    user_input = st.sidebar.text_input("", "Suzanne Collins")
    dg = df.loc[df["Author"] == user_input].sort_values(by=['avg_ratings'], ascending= False).head(1)
    
    st.sidebar.write("Want to read her/his highest rated book?")
    for i in range(1, 2):
        cols = st.sidebar.beta_columns(1)
        cols[0].table(dg['Setting']) #use_column_width=True)
    for i in range(1, 2):
        cols = st.sidebar.beta_columns(1)
        cols[0].table(dg['Series']) #use_column_width=True)

    isbn = 3789120022

    link = f'[BUY YOUR FAVORITE BOOK FROM {user_input} ON AMAZON] (https://www.amazon.de/s?k={isbn}&__mk_de_DE=ÅMÅŽÕÑ&ref=nb_sb_noss)'
    st.sidebar.markdown(link, unsafe_allow_html=True)        
        #cols[1].write(f'{i * i}')
        #cols[2].write(f'{i * i * i}')
        #cols[3].write('x' * i)
    
    #for i in range(1, 2):
    #    cols = st.sidebar.beta_columns(1)
    #    cols[0].image(avg_rating_img, use_column_width=True)
    #for i in range(1, 2):
    #    cols = st.sidebar.beta_columns(1)
    #    cols[0].image(mean_norm_rating_img, use_column_width=True)
    #for i in range(1, 2):
    #    cols = st.sidebar.beta_columns(1)
    #    cols[0].image(all_fitted_dist_img, use_column_width=True)
    #for i in range(1, 2):
    #    cols = st.sidebar.beta_columns(1)
    #    cols[0].image(avg_rat_burr_img, use_column_width=True)


    # Show the 2 scatterplots of pages and num_ratings / num_awards
    st.markdown("""Got a long attention span? No? We thought so too, the data supports that too. You are all good :D!
People love books that are not too long, it just keeps us interested. Facts! The graphs do not lie, that is why we award such books greatly too!""")

    df_s = df.dropna(subset=['Num_pages', 'number_of_ratings'])
    plt.figure(figsize=(20,10))
    x1  = df_s['Num_pages'].astype(int)
    x2 = df_s['Num_pages'].astype(int)

    y1 = df_s['number_of_ratings'].astype(int)
    y2 = df_s['award_count'].astype(int)

    plt.subplot(1,2,1)
    plt.scatter(x1, y1, color='blue')
    plt.xlabel("Number of pages")
    plt.ylabel("Number of ratings")

    plt.subplot(1,2,2)
    plt.scatter(x2, y2, color='red')
    plt.xlabel("Number of pages")
    plt.ylabel("Number of awards")

    plt.tight_layout()
    st.pyplot(plt)


    avg_rating_img = Image.open("./IMG/average_rating.png")
    mean_norm_rating_img = Image.open("./IMG/mean_normalized_rating.png")
    all_fitted_dist_img = Image.open("./IMG/all_fitted_dist.png")
    avg_rat_burr_img = Image.open("./IMG/avg_rat_burr.png")

    st.header("You are a curious elf and want to dive even deeper?")
    for i in range(1, 2):
        cols = st.beta_columns(4)
        cols[0].image(avg_rating_img, use_column_width=True)
        cols[1].image(mean_norm_rating_img, use_column_width=True)
        cols[2].image(all_fitted_dist_img, use_column_width=True)
        cols[3].image(avg_rat_burr_img, use_column_width=True)

    for i in range(1, 2):
        cols = st.beta_columns(4)
        cols[0].write('Histogram for average rating with supporting boxplot.', use_column_width=True)
        cols[1].write("Boxplot and histogram for min-max normalized ratings.", use_column_width=True)
        cols[2].write('Several possible fits were tested on the distribution graphs.', use_column_width=True)
        cols[3].write('The burr distribution was the best fit.', use_column_width=True)

    st.markdown("![The best EPIC fantasy book](https://media.giphy.com/media/Y1vq7S47uaH1hqYhf6/source.gif)")

    # AWARDS

    st.markdown('Like the protagonists in their stories, the books fight relentlessly for honor and profit')
    # Show the number of awards with the amount of books in it
    dz = df['award_count'].value_counts()
    dw= dz.sort_index()
    dw = pd.DataFrame(dw)
    dw.reset_index(inplace=True)
    dw = dw.rename(columns = {'index':'award_count', 'award_count': 'num_books'})
    sns.set_context('talk')
    sns.set_style('whitegrid')
    plt.figure(figsize=(14, 8))
    ax= sns.barplot(y=dw['num_books'],x=dw["award_count"])
    ax.set_yscale("log")
    for p in ax.patches:
        ax.annotate(format(int(p.get_height())), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')
    ax= plt.xlabel('Number of awards')
    ax= plt.ylabel('Number of books')

    st.pyplot(plt)

    # Show a list of the TOP TEN awarded books 
    st.subheader("And the most awards for best EPIC fantasy book go to...")
    st.markdown("![The best EPIC fantasy book](https://media.giphy.com/media/c35lTs1WBrqJW/source.gif)")
    df_sorted = df.sort_values(by='award_count', ascending=False).head(10)
    st.dataframe(df_sorted['Title'])

    # Books with more awards than X
    #st.markdown("")
    s2_min = int(df.award_count.min())
    s2_max = int(df.award_count.max())
    threshold = st.slider("Just show books with more awards than:", s2_min, s2_max)
    filtered = df[df["award_count"] >= threshold]
    if st.checkbox("Select by awards"):
        placeholder = st.empty()
        placeholder.dataframe(filtered[columns_to_show])
    #st.dataframe(filtered[columns_to_show])

    # More Ratings = More Awards? 
    st.subheader("Here you can see the number of ratings with the number of awards")
    sns.set_context('talk')
    sns.set_style('whitegrid')
    sns.set_palette(['purple'])
    plt.figure(figsize=(20, 10))
    ax= sns.regplot(data=df, y="number_of_ratings", x="award_count")
    ax= plt.ylabel("Number of awards")
    ax= plt.xlabel("Number of ratings")

    st.pyplot(plt)


else:
    st.markdown("![You shall not pass](https://media.giphy.com/media/nlcJ4MkoG3Tri/giphy.gif)") 
    st.title("You shall not pass")


# Interactive Genre List

# 8 Visualize the awards distribution in a barplot with a list of books shown when clicked on a bar

