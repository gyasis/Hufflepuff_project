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


# In[21]:


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

    return df

preprocessing()


# In[ ]:


# TopGif
st.markdown("![The world of fantasy](https://media.giphy.com/media/3o7bu7BKhyOJt36jp6/source.gif)")

st.title("Welcome to the world of fantasy")


# In[28]:


# Show our data

st.header("The data of the 1000 best epic fantasy books")

st.markdown("""Some nice text about fantasy books""")

#df = pd.read_csv("example.csv")
if st.button("Show me the data."):
    st.dataframe(df)

st.markdown("If you want to display only few columns, you can chose here:")
columns_to_show = st.multiselect('', df.columns)

st.markdown("Just show books with a rating higher than:")
threshold = st.slider("", 1, 10)
filtered = df[df["minmax_norm_ratings"] >= threshold]
st.dataframe(filtered[columns_to_show])

st.markdown("Just show books with more awards than:")
threshold = st.slider("", 0, 50)
filtered = df[df["award_count"] >= threshold]
st.dataframe(filtered[columns_to_show])


# Show the 2 scatterplots of pages and num_ratings / num_awards
st.subheader("A 2D scatterplot with pages on the x-axis and num_ratings / num_awards on the y-axis")

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

st.subheader("And the award for best EPIC fantasy book goes to...")
st.markdown("![The world of fantasy](https://media.giphy.com/media/3o7bu7BKhyOJt36jp6/source.gif)")
df_sorted = df.sort_values(by='award_count', ascending=False).head(10)
st.dataframe(df_sorted['Title'])


# In[ ]:


# Interactive Genre List


# In[ ]:


# 8 Visualize the awards distribution in a barplot with a list of books shown when clicked on a bar

