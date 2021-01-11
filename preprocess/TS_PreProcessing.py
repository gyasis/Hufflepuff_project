#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Option 1:


# In[1]:


import pandas as pd

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
    
    df['avg_ratings_min_max_norm'] = 1 + (df['avg_ratings'] - df.avg_ratings.min()) / (df.avg_ratings.max()-df.avg_ratings.min()) *9

    df['avg_ratings_mean'] = 1 + (df['avg_ratings'] - df.avg_ratings.mean()) / (df.avg_ratings.max()-df.avg_ratings.min()) *9
    
    print(df)

preprocessing()


# In[ ]:


# Option 2: (Just want to check which method is faster for counting the awards)


# In[112]:


import pandas as pd
#df = pd.read_csv('Time.csv')

df = pd.DataFrame([
  { 'id': 1, 'score': 4.1, 'awards': 'Hello, Hello, Hello' },
  { 'id': 2, 'score': 4.5, 'awards': 'Hello, Hello' },
  { 'id': 3, 'score': 4.5, 'awards': 'Hello' },
  { 'id': 4, 'score': 4.6, 'awards': 'Hello, Hello, Hello, Hello, Hello' }, 
  { 'id': 5, 'score': 3.2, 'awards': 'Hello, Hello, Hello, Hello' }, 
  { 'id': 6, 'score': 2.8, 'awards': 'Hello, Hello, Hello' }])

df.awards = df.awards.str.split(',', expand=False)
award_count = []
for i in df.index:
    award_count.append(len(df.iloc[i]['awards']))
df['Award Count'] = award_count
print(df)

