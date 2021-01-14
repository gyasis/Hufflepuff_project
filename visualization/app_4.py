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

def preprocessing():
    df = pd.read_csv("EpicFantasy_2.csv")
    df.rename(columns={'name':'Title', 'OriginalPDate':'Publishdate', 'author':'Author'}, inplace=True)
    
    # cleaning the Title
    df['Title'] = df['Title'].str.strip()
    
    # Calculating number of awards
    df['Awards'] = df['Awards'].str.strip('[]')
    df['Awards'] = df['Awards'].replace(r'^\s*$', np.NaN, regex=True)
    df['Awards_x'] = df.Awards.str.split(',')
    df['award_count'] = df['Awards_x'].str.len()
    df['award_count'] = df['award_count'].fillna(0).astype(int)
    df = df.drop(['Awards_x'], axis= 1)
    
    # Cleaning author column from paranthesis text
    df['Author'] = df.Author.str.replace(r"\(.*Goodreads Author\)","")

    # cleaning number of ratings
    df.rename(columns={'number of ratings': 'number_of_ratings'}, inplace=True)
    df['number_of_ratings'] = df['number_of_ratings'].str.replace('[,]',"").str.replace('[^0-9]',"0").fillna(0).astype(int)

    # cleaning avg_rat
    #df['avg_ratings'] = df.avg_ratings.str.replace('[^0-9.]',"")
    df['avg_ratings'] = df['avg_ratings'].replace(r'^\s*$', np.NaN, regex=True).astype(float)
    
    # cleaning Genre
    df['Genre'] = df['Genre'].str.strip('[]')
    
    # cleaning pages
    df['Num_pages'] = df['Num_pages'].astype(str)
    df['Num_pages'] = df.Num_pages.str.replace('[^0-9]',"")
    for i in df.index:
        if len(df.iloc[i]['Num_pages']) >= 7 or len(df.iloc[i]['Num_pages']) <= 1:
            df.at[i, 'Num_pages'] = np.NaN
    df['Num_pages'] = df['Num_pages'].replace(r'^\s*$', np.NaN, regex=True)
    df['Num_pages'] = pd.to_numeric(df['Num_pages'])
    
    # cleaning publish year
    df.rename(columns={' Publishdate': 'Publishdate'}, inplace=True)
    df['Publishdate'] = df.Publishdate.str.replace('[^0-9-]',"")
    df['Publishdate'] = df['Publishdate'].replace(r'^\s*$', np.NaN, regex=True)
    df['original_publish_year'] = df['Publishdate'].str[-4:]
    df = df.drop(['Publishdate'], axis= 1)

    # Calculating normalized ratings
    df['minmax_norm_ratings'] = 1 + (df['avg_ratings'] - df.avg_ratings.min()) / (df.avg_ratings.max()-df.avg_ratings.min()) *9
    df['mean_norm_ratings'] = 1 + ((df['avg_ratings'] - df.avg_ratings.mean()) / (df.avg_ratings.max()-df.avg_ratings.min())) *9

    # delete last 7 rows    
    df.drop(df.tail(7).index, inplace = True) 

    # delete unnecessary columns
    df.drop(['Unnamed: 0', 'id'], axis = 1, inplace = True)

    return df

preprocessing()

df = preprocessing()


# In[ ]:


# TITLE
st.markdown("![The world of fantasy](https://media.giphy.com/media/3o7bu7BKhyOJt36jp6/source.gif)")

st.title("Welcome to the world of fantasy")


# In[45]:


# Show our data

st.header("The data of the 1000 best epic fantasy books")

st.markdown("""Take a look at the infos""")

st.sidebar.header("Visualization Selector")
#st.sidebar.text("Writing on the sidebar!")
st.sidebar.markdown("""Tm!""")


#df = pd.read_csv("example.csv")
placeholder = st.empty()
if st.sidebar.checkbox("First select to see the data"):
    placeholder.dataframe(df)
    
#if st.sidebar.button("Show me the data."):
#    st.dataframe(df)

# Choose columns to show
#st.sidebar.markdown("")
columns_to_show = st.sidebar.multiselect('If you want to display only few columns, you can chose here:', df.columns)

# Books with rating higher than X
#st.markdown("")
s1_min = int(df.minmax_norm_ratings.min())
s1_max = int(df.minmax_norm_ratings.max())
threshold = st.sidebar.slider("Just show books with a rating higher than:", s1_min, s1_max)
filtered = df[df["minmax_norm_ratings"] >= threshold]
st.dataframe(filtered[columns_to_show])


# Show the 2 scatterplots of pages and num_ratings / num_awards
st.subheader("What can we say about these nice scatterplots")

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
st.subheader("And the most awards for best EPIC fantasy book goes to...")
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
    placeholder.dataframe(filtered[columns_to_show])
#st.dataframe(filtered[columns_to_show])

# More Ratings = More Awards? 
st.subheader("Here you can see the number of ratings with the number of awards")
sns.set_context('talk')
sns.set_style('whitegrid')
sns.set_palette(['purple'])
plt.figure(figsize=(20, 10))
ax= sns.regplot(data=preprocessing(), y="number_of_ratings", x="award_count")
ax= plt.ylabel("Number of awards")
ax= plt.xlabel("Number of ratings")

st.pyplot(plt)


# In[ ]:





# In[ ]:


# Interactive Genre List

# 8 Visualize the awards distribution in a barplot with a list of books shown when clicked on a bar

