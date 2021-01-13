#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

def preprocessing():
    #df = pd.read_csv(csv_path)

    df = pd.DataFrame([
      { 'author': 'abc', 'avg_ratings': 4.1, 'awards': 'Hello, Hello, Hello', 'original_publish_year': 2010},
      { 'author': 'def', 'avg_ratings': 2.5, 'awards': 'Hello, Hello', 'original_publish_year': 2012},
      { 'author': 'abc', 'avg_ratings': 4.5, 'awards': 'Hello', 'original_publish_year': 2010},
      { 'author': 'def', 'avg_ratings': 3.3, 'awards': 'Hello, Hello, Hello, Hello, Hello', 'original_publish_year': 2012}, 
      { 'author': 'abc', 'avg_ratings': 3.8, 'awards': 'Hello, Hello, Hello, Hello', 'original_publish_year': 2010}, 
      { 'author': 'obc', 'avg_ratings': 2.8, 'awards': 'Hello, Hello, Hello', 'original_publish_year': 2012}])

    df['awards'] = df.awards.str.split(',', expand=False)
    df['award count'] = df['awards'].str.len()
    
    df['minmax_norm_ratings'] = 1 + (df['avg_ratings'] - df.avg_ratings.min()) / (df.avg_ratings.max()-df.avg_ratings.min()) *9

    df['avg_ratings_mean'] = 1 + (df['avg_ratings'] - df.avg_ratings.mean()) / (df.avg_ratings.max()-df.avg_ratings.min()) *9

    return df

preprocessing()


# In[13]:


def analyse_highest_book(df, a):
    dp = df.groupby(df.original_publish_year).agg({'minmax_norm_ratings': np.mean})
    dp = dp.rename(columns={'minmax_norm_ratings': 'minmax_norm_ratings_mean'})
    author_group = df.loc[df['author'] == a]
    book_max_rating = author_group.iloc[author_group['minmax_norm_ratings'].argmax()]['awards']
    print(dp)
    return book_max_rating

print(analyse_highest_book(preprocessing(), 'abc'))

