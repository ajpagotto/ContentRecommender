
# Content Based Movie Recommendations
Andrea Pagotto

## Introduction

This notebook will explain implementation and experimentations done on the Movie Lens dataset, downloaded through Kaggle ("The Movies Dataset"). The dataset consists of several different csv files, containing different types of information about the movies including, identification numbers for imdb and tmdb, metadata about the cast and crew, descriptions of the movies, keywords and more. The main features of the data focussed on in this notebook will be the text data: descriptions, keywords, and metadata. Using this data, as well as data about user ratings for each movie, recommendation techniques will be experimented with. 

The main recommendation approaches being assessed are:
- content based recommendations to identify similar movies
- content based recommendations for a particular user
- classic collaborative filtering
- a hybrid approach

First, the notebook will show how to load and format data to prepare for use in similarity comparisons. Next, the notebook will show how to represent text features in different ways including tfidf representations and word embeddings. Finally, this notebook will show how to use these features to generate movie similarity rankings for different types of recommenders.

Note: This notebook can be downloaded but not run directly without dowloading the GloVe vectors.

For further explanations on background and results, please see: text-feature-representation.pdf

## Loading the Datasets

First will load the required pakages.

Glove word embeddings must be dowloaded from: https://nlp.stanford.edu/projects/glove/


```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
from scipy import spatial
from scipy.spatial.distance import pdist, squareform
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# import custom functions
import nlpProjectFunctions
from nlpProjectFunctions import get_director, convert_int


import warnings; warnings.simplefilter('ignore')
```

### Loading Data and Formatting with Pandas
Generate a data frame from the full dataset, then restrict the size to only the movies present in the smaller dataset.


```python
# Load all datasets
md = pd. read_csv('the-movies-dataset/movies_metadata.csv')
links_small = pd.read_csv('the-movies-dataset/links_small.csv')
credits = pd.read_csv('the-movies-dataset/credits.csv')
keywords = pd.read_csv('the-movies-dataset/keywords.csv')
ratings = pd.read_csv('the-movies-dataset/ratings_small.csv')
# Create a mapping from the small links dataset
id_map = pd.read_csv('the-movies-dataset/links_small.csv')[['movieId', 'tmdbId']]
```

Format the full dataset into a panda dataframe.


```python
# reformat the genres, id and year columns, drop some indices
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')

# load key words and credits data into the md
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')

md.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>...</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>year</th>
      <th>cast</th>
      <th>crew</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[Animation, Comedy, Family]</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
      <td>[{'cast_id': 14, 'character': 'Woody (voice)',...</td>
      <td>[{'credit_id': '52fe4284c3a36847f8024f49', 'de...</td>
      <td>[{'id': 931, 'name': 'jealousy'}, {'id': 4290,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[Adventure, Fantasy, Family]</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': 'Alan Parrish', '...</td>
      <td>[{'credit_id': '52fe44bfc3a36847f80a7cd1', 'de...</td>
      <td>[{'id': 10090, 'name': 'board game'}, {'id': 1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>{'id': 119050, 'name': 'Grumpy Old Men Collect...</td>
      <td>0</td>
      <td>[Romance, Comedy]</td>
      <td>NaN</td>
      <td>15602</td>
      <td>tt0113228</td>
      <td>en</td>
      <td>Grumpier Old Men</td>
      <td>A family wedding reignites the ancient feud be...</td>
      <td>...</td>
      <td>Released</td>
      <td>Still Yelling. Still Fighting. Still Ready for...</td>
      <td>Grumpier Old Men</td>
      <td>False</td>
      <td>6.5</td>
      <td>92.0</td>
      <td>1995</td>
      <td>[{'cast_id': 2, 'character': 'Max Goldman', 'c...</td>
      <td>[{'credit_id': '52fe466a9251416c75077a89', 'de...</td>
      <td>[{'id': 1495, 'name': 'fishing'}, {'id': 12392...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>NaN</td>
      <td>16000000</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>NaN</td>
      <td>31357</td>
      <td>tt0114885</td>
      <td>en</td>
      <td>Waiting to Exhale</td>
      <td>Cheated on, mistreated and stepped on, the wom...</td>
      <td>...</td>
      <td>Released</td>
      <td>Friends are the people who let you be yourself...</td>
      <td>Waiting to Exhale</td>
      <td>False</td>
      <td>6.1</td>
      <td>34.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': "Savannah 'Vannah...</td>
      <td>[{'credit_id': '52fe44779251416c91011acb', 'de...</td>
      <td>[{'id': 818, 'name': 'based on novel'}, {'id':...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>{'id': 96871, 'name': 'Father of the Bride Col...</td>
      <td>0</td>
      <td>[Comedy]</td>
      <td>NaN</td>
      <td>11862</td>
      <td>tt0113041</td>
      <td>en</td>
      <td>Father of the Bride Part II</td>
      <td>Just when George Banks has recovered from his ...</td>
      <td>...</td>
      <td>Released</td>
      <td>Just When His World Is Back To Normal... He's ...</td>
      <td>Father of the Bride Part II</td>
      <td>False</td>
      <td>5.7</td>
      <td>173.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': 'George Banks', '...</td>
      <td>[{'credit_id': '52fe44959251416c75039ed7', 'de...</td>
      <td>[{'id': 1009, 'name': 'baby'}, {'id': 1599, 'n...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



Create a subset of the data set using only the movies in the smaller dataset.


```python
print(md[md['title']=='Mean Girls'].index)
md[md['title']=='Mean Girls']
```

    Int64Index([7353], dtype='int64')





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>...</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>year</th>
      <th>cast</th>
      <th>crew</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7353</th>
      <td>False</td>
      <td>{'id': 99606, 'name': 'Mean Girls Collection',...</td>
      <td>17000000</td>
      <td>[Comedy]</td>
      <td>http://www.meangirls.com/</td>
      <td>10625</td>
      <td>tt0377092</td>
      <td>en</td>
      <td>Mean Girls</td>
      <td>Cady Heron is a hit with The Plastics, the A-l...</td>
      <td>...</td>
      <td>Released</td>
      <td>Welcome to girl world.</td>
      <td>Mean Girls</td>
      <td>False</td>
      <td>6.9</td>
      <td>2401.0</td>
      <td>2004</td>
      <td>[{'cast_id': 9, 'character': 'Cady Heron', 'cr...</td>
      <td>[{'credit_id': '5635ec3092514129fe00c2f5', 'de...</td>
      <td>[{'id': 5248, 'name': 'female friendship'}, {'...</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 28 columns</p>
</div>




```python
# to use a subset of the data, restrict the full data to only 
# the movies in the smaller dataset
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

# for each id in the links dataset, extract that indice
smd = md[md['id'].isin(links_small)]
smd.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>...</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>year</th>
      <th>cast</th>
      <th>crew</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[Animation, Comedy, Family]</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
      <td>[{'cast_id': 14, 'character': 'Woody (voice)',...</td>
      <td>[{'credit_id': '52fe4284c3a36847f8024f49', 'de...</td>
      <td>[{'id': 931, 'name': 'jealousy'}, {'id': 4290,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[Adventure, Fantasy, Family]</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': 'Alan Parrish', '...</td>
      <td>[{'credit_id': '52fe44bfc3a36847f80a7cd1', 'de...</td>
      <td>[{'id': 10090, 'name': 'board game'}, {'id': 1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>{'id': 119050, 'name': 'Grumpy Old Men Collect...</td>
      <td>0</td>
      <td>[Romance, Comedy]</td>
      <td>NaN</td>
      <td>15602</td>
      <td>tt0113228</td>
      <td>en</td>
      <td>Grumpier Old Men</td>
      <td>A family wedding reignites the ancient feud be...</td>
      <td>...</td>
      <td>Released</td>
      <td>Still Yelling. Still Fighting. Still Ready for...</td>
      <td>Grumpier Old Men</td>
      <td>False</td>
      <td>6.5</td>
      <td>92.0</td>
      <td>1995</td>
      <td>[{'cast_id': 2, 'character': 'Max Goldman', 'c...</td>
      <td>[{'credit_id': '52fe466a9251416c75077a89', 'de...</td>
      <td>[{'id': 1495, 'name': 'fishing'}, {'id': 12392...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>NaN</td>
      <td>16000000</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>NaN</td>
      <td>31357</td>
      <td>tt0114885</td>
      <td>en</td>
      <td>Waiting to Exhale</td>
      <td>Cheated on, mistreated and stepped on, the wom...</td>
      <td>...</td>
      <td>Released</td>
      <td>Friends are the people who let you be yourself...</td>
      <td>Waiting to Exhale</td>
      <td>False</td>
      <td>6.1</td>
      <td>34.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': "Savannah 'Vannah...</td>
      <td>[{'credit_id': '52fe44779251416c91011acb', 'de...</td>
      <td>[{'id': 818, 'name': 'based on novel'}, {'id':...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>{'id': 96871, 'name': 'Father of the Bride Col...</td>
      <td>0</td>
      <td>[Comedy]</td>
      <td>NaN</td>
      <td>11862</td>
      <td>tt0113041</td>
      <td>en</td>
      <td>Father of the Bride Part II</td>
      <td>Just when George Banks has recovered from his ...</td>
      <td>...</td>
      <td>Released</td>
      <td>Just When His World Is Back To Normal... He's ...</td>
      <td>Father of the Bride Part II</td>
      <td>False</td>
      <td>5.7</td>
      <td>173.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': 'George Banks', '...</td>
      <td>[{'credit_id': '52fe44959251416c75039ed7', 'de...</td>
      <td>[{'id': 1009, 'name': 'baby'}, {'id': 1599, 'n...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



### Prepare Data for Description Based Recommender

First format the text in the text-containing columns.


```python
# Process the tagline column, and create a description column
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

smd = smd.reset_index() 
smd.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>...</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>year</th>
      <th>cast</th>
      <th>crew</th>
      <th>keywords</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[Animation, Comedy, Family]</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>...</td>
      <td></td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
      <td>[{'cast_id': 14, 'character': 'Woody (voice)',...</td>
      <td>[{'credit_id': '52fe4284c3a36847f8024f49', 'de...</td>
      <td>[{'id': 931, 'name': 'jealousy'}, {'id': 4290,...</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[Adventure, Fantasy, Family]</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>...</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': 'Alan Parrish', '...</td>
      <td>[{'credit_id': '52fe44bfc3a36847f80a7cd1', 'de...</td>
      <td>[{'id': 10090, 'name': 'board game'}, {'id': 1...</td>
      <td>When siblings Judy and Peter discover an encha...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>False</td>
      <td>{'id': 119050, 'name': 'Grumpy Old Men Collect...</td>
      <td>0</td>
      <td>[Romance, Comedy]</td>
      <td>NaN</td>
      <td>15602</td>
      <td>tt0113228</td>
      <td>en</td>
      <td>Grumpier Old Men</td>
      <td>...</td>
      <td>Still Yelling. Still Fighting. Still Ready for...</td>
      <td>Grumpier Old Men</td>
      <td>False</td>
      <td>6.5</td>
      <td>92.0</td>
      <td>1995</td>
      <td>[{'cast_id': 2, 'character': 'Max Goldman', 'c...</td>
      <td>[{'credit_id': '52fe466a9251416c75077a89', 'de...</td>
      <td>[{'id': 1495, 'name': 'fishing'}, {'id': 12392...</td>
      <td>A family wedding reignites the ancient feud be...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>False</td>
      <td>NaN</td>
      <td>16000000</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>NaN</td>
      <td>31357</td>
      <td>tt0114885</td>
      <td>en</td>
      <td>Waiting to Exhale</td>
      <td>...</td>
      <td>Friends are the people who let you be yourself...</td>
      <td>Waiting to Exhale</td>
      <td>False</td>
      <td>6.1</td>
      <td>34.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': "Savannah 'Vannah...</td>
      <td>[{'credit_id': '52fe44779251416c91011acb', 'de...</td>
      <td>[{'id': 818, 'name': 'based on novel'}, {'id':...</td>
      <td>Cheated on, mistreated and stepped on, the wom...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>False</td>
      <td>{'id': 96871, 'name': 'Father of the Bride Col...</td>
      <td>0</td>
      <td>[Comedy]</td>
      <td>NaN</td>
      <td>11862</td>
      <td>tt0113041</td>
      <td>en</td>
      <td>Father of the Bride Part II</td>
      <td>...</td>
      <td>Just When His World Is Back To Normal... He's ...</td>
      <td>Father of the Bride Part II</td>
      <td>False</td>
      <td>5.7</td>
      <td>173.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': 'George Banks', '...</td>
      <td>[{'credit_id': '52fe44959251416c75039ed7', 'de...</td>
      <td>[{'id': 1009, 'name': 'baby'}, {'id': 1599, 'n...</td>
      <td>Just when George Banks has recovered from his ...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
print(smd[smd['title']=='Mean Girls'].index)
# note the index for this movie has changed
```

    Int64Index([5207], dtype='int64')


### Preparing Data for Metadata Recommender
Format the columns containing metadata to be able to extract words for processing.


```python
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['cast'] = smd['cast'].apply(literal_eval)

smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

smd['director'] = smd['crew'].apply(get_director)
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x, x])
```

#### Processing Keywords and Metadata into One
Creating a new column to contain all the relevant data from the metadata as a bag of features, called "soup".


```python
# requires s that is defined below
def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

# s is a reduced list of keywords, with only the eywords that occur more than once
s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
print(s[:5])
s = s[s > 1]

# create a stemmer 
stemmer = SnowballStemmer('english')

# process words using filter and stemmer
smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# create a soup column to contain all processed metadata words
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

smd = smd.reset_index()
```

    independent film        610
    woman director          550
    murder                  399
    duringcreditsstinger    327
    based on novel          318
    Name: keyword, dtype: int64



```python
smd.shape
```




    (9219, 35)



### Create Mappings

Create mappings to access different information given title or id. Note there are three different ids associated with each movie: its index in the dataset, the imdb id, and the tmdb id. The ratings dataset only has the imdb and tmdb ids, no titles, so it is necessary to be able to access movies from each of these ids, so mappings will be made to allow this.


```python
# Create a title to index mapping, so indices for 
# a title can be found.
# Note smd.index is just its position in the small dataset
# smd.index will be the index of this item in arrays
titles = smd['title']
title2index = pd.Series(smd.index, index=titles)
#print(indices.head())

# indices can be accessed from the title, but note it
# returns multiple items when the titles occurs more 
# than once like in the movie dracula
print("The Addams Family Index:")
print(title2index.loc['The Addams Family'])
print("Dracula Index:")
print(title2index.loc['Dracula'])
```

    The Addams Family Index:
    1701
    Dracula Index:
    title
    Dracula    1100
    Dracula    2135
    Dracula    4797
    dtype: int64


Because the titles can occur more than once in the dataset resulting in inconsistent return values, the movies will instead be access by their ids, as these ids are unique within the dataset. To do this more mappings will be made.


```python
# Check the ids for the previous movies
print(smd[smd['title'] == 'Dracula']['id'])
print(smd[smd['title'] == 'The Addams Family']['id'])
```

    1100     6114
    2135      138
    4797    33521
    Name: id, dtype: int64
    1701    2907
    Name: id, dtype: int64



```python
# Map the ids to the index same as done with titles
# 'id' is the tmdb ids in metadata and small links data
# this id can be mapped to index, which will be the index of
# the movie in an array, in later functions
ids = smd['id']
id2index = pd.Series(smd.index, index=ids)
print("Movie 6114 (Dracula) Index:")
print(id2index.loc[6114])
print("Movie 2907 (Addams Family) Index:")
print(id2index.loc[2907])
```

    Movie 6114 (Dracula) Index:
    1100
    Movie 2907 (Addams Family) Index:
    1701


Next because the ratings dataset contains only the "movie id" ids, create mappings between the currently used tmdb ids and these ids. The small links dataset contains all three ids, and the metadata datset only contains the tmdb and imdb ids. The column 'id' in the metadata datset corresponds to 'tmdb id' in the small links dataset.


```python
# id map will map the movie id needed for the 
# ratings data to the tmdbId, and title
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']

# The index is currently being set to title, however do to non unique titles
# an alternative version of the mapping will be made
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
#id_map = id_map.set_index('tmdbId')

print(id_map.head())
```

                                 movieId     id
    title                                      
    Toy Story                          1    862
    Jumanji                            2   8844
    Grumpier Old Men                   3  15602
    Waiting to Exhale                  4  31357
    Father of the Bride Part II        5  11862



```python
# example how to access a title from an index with this mapping
title = id_map[id_map['id']==862].index
print(id_map[id_map['id']==862].index)

# how to access ids for a certain title
print(id_map.loc[title])
```

    Index(['Toy Story'], dtype='object', name='title')
               movieId   id
    title                  
    Toy Story        1  862


The result of this mapping will print the tmdb id and movie id (in the ratings dataset) for a given title). Next we will make a mapping that is indexed by the movie id instead.


```python
movieid_map = id_map.set_index('movieId')
print(movieid_map.head())
print(type(movieid_map))
# retrieve a tmdb id from movieId 1339
print("Tmdb id for movie id 1339:")
print(movieid_map.loc[1339]['id'])

tmdb_map = id_map.set_index('id')
print(tmdb_map.head())

# retrieve a movieId from a tmdb id 6114
print("movie id for tmdb id 6114:")
print(tmdb_map.loc[6114]['movieId'])

# how to use this map in the reverse direction
print("Tmdb id for movie id 1339:")
print(tmdb_map[tmdb_map['movieId']==1339].index)
```

                id
    movieId       
    1          862
    2         8844
    3        15602
    4        31357
    5        11862
    <class 'pandas.core.frame.DataFrame'>
    Tmdb id for movie id 1339:
    6114.0
             movieId
    id              
    862.0          1
    8844.0         2
    15602.0        3
    31357.0        4
    11862.0        5
    movie id for tmdb id 6114:
    1339
    Tmdb id for movie id 1339:
    Float64Index([6114.0], dtype='float64', name='id')



```python
movieIds = tmdb_map['movieId']
print(movieIds.head())

# a way to check if a movie id is in the dataset
print(55207 in movieid_map.index)
```

    id
    862.0      1
    8844.0     2
    15602.0    3
    31357.0    4
    11862.0    5
    Name: movieId, dtype: int64
    False


These mappings will be used in the upcoming functions to retrieve information about the movies as needed

### Loading Data with User Ratings

This code will use the ratings dataset, and use a reader to read it into a dataset format for use in the recommender functions. Also, it will use a built in function to generate splits for cross validation.


```python
reader = Reader()
ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)
```

## Word Representations

This section will implement and experiment with word represetations to prepare for use in generating features from the text data. In particular, this section will focus on representing words with word embeddings.

### Generating Tokens

The first step in representing text, is to tokenize the texts. This process will create a list of "tokens" which in this case will just be words. To do this, will will create a custom tokenization function that will apply all necessary processing to the raw text descriptions. This function is shown in the following code. This function will tokenize using the regex expression for words, then it will transform all the words to lower case and remove stop words, as well as single characters. The last step of the preprocessing will remove numbers, to result in only words remaining for the word embeddings.


```python
def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text)
    mod = [w.lower() for w in tokenized if len(w) > 1 and w.lower() not in stopwords.words('english')]
    words = [w for w in mod if not w.isnumeric()]
    return  words

# apply the tokenizer to the descriptions
tokens = smd['description'].apply(tokenize)

# print a sample of a tokenized description and the associated movie title
print(titles[1])
print(tokens[1])
print(len(tokens))
```

    Jumanji
    ['siblings', 'judy', 'peter', 'discover', 'enchanted', 'board', 'game', 'opens', 'door', 'magical', 'world', 'unwittingly', 'invite', 'alan', 'adult', 'trapped', 'inside', 'game', 'years', 'living', 'room', 'alan', 'hope', 'freedom', 'finish', 'game', 'proves', 'risky', 'three', 'find', 'running', 'giant', 'rhinoceroses', 'evil', 'monkeys', 'terrifying', 'creatures', 'roll', 'dice', 'unleash', 'excitement']
    9219


### Implement Word Embeddings - GloVe

Once the text is in tokenized form, word embeddings can be applied. In order to do this, first a dictionary of word embeddings must be loaded from a text file and processed into the right format to be able to look up words as a python dictionary. Pre-trained word embeddings will be loaded that were trained useing the GloVe algorithm. After these vectors are loaded they will be evaulated to assess that they are produce meaningful results. This evalution procedure can be used as part of the assessement towards determining which wod embeddings should be used for a particular application, and comparisons can be done with other word vectors, such as google word2vec.

#### Loading Pretrained Vectors

The first step in the procedure is to load the glove text file and format it into dictionaries. Mappings are made between word indices in both directions, and also between words and their embedding vector representation.


```python
# get vocab and create dict with vocab indexes
glove_file = 'glove.6B.50d.txt'
with open(glove_file, 'r') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}  
```

The next code will create the mapping between the words and their embedding as a numpy array.


```python
# Dictionary to create word embeddings 
embeddings_index = {}

# using the glove text file once again to load the vectors
f = open(glove_file)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# print a sample of one array representing a word 
print("The word 'hello':")
print(embeddings_index['hello'])
```

    Found 400000 word vectors.
    The word 'hello':
    [-0.38497001  0.80092001  0.064106   -0.28354999 -0.026759   -0.34531999
     -0.64253002 -0.11729    -0.33256999  0.55242997 -0.087813    0.90350002
      0.47102001  0.56656998  0.69849998 -0.35229    -0.86541998  0.90573001
      0.03576    -0.071705   -0.12327     0.54922998  0.47005001  0.35572001
      1.26110005 -0.67580998 -0.94983     0.68665999  0.38710001 -1.34920001
      0.63511997  0.46416    -0.48813999  0.83827001 -0.92460001 -0.33722001
      0.53741002 -1.06159997 -0.081403   -0.67110997  0.30923    -0.39230001
     -0.55001998 -0.68826997  0.58048999 -0.11626     0.013139   -0.57653999
      0.048833    0.67203999]



```python
vector_dim = len(embeddings_index['hello'])
print("Dimensions of the embeddings:")
print(vector_dim)
```

    Dimensions of the embeddings:
    50


It can be seen the word hello is represented as a 50 dimension numpy array. Loading a different file can change this, as the GloVe pretrained embeddings are available in various dimensions.

#### Experimenting with Word Representations

Before proceding with the process of generating document vectors, first we will inspect the results of the loaded glove word embeddings. We can compare the similarities between words using the cosine distance and make sure they are being properly represented.


```python
# Store all the word embeddings in a numpy array
# Each embedding can still be access from the array
# using the word index in the vocab mapping dictionary
W = np.zeros((vocab_size, vector_dim))
for word, v in embeddings_index.items():
    if word == '<unk>':
        continue
    # the word vector is stored at its index in vocab
    W[vocab[word], :] = v
```

Next we will create a normalized version of this embedding matrix, which is the equivalient of mapping each vector to its projection on a unit hypersphere, ie. reducing the length of each vector to one, in order to only have variance between words in the angles of the vectors. This will come in handy when measuring cosine distances between words, as will be explained in the following sections.


```python
# normalize each word vector to unit variance
W_norm = np.zeros(W.shape)
d = (np.sum(W ** 2, 1) ** (0.5))
W_norm = (W.T / d).T
```

These GloVe vectors are evaluated with some benchmark testing function provided by the creators of GloVe. This can give an indication that the GloVe vectors being used are properly representing words as they should. The evaluation is done by using the GloVe vectors to guess the answers to questions in various files, using lists of given words. The accuracy of the predicting the correct answer is shown, and higher accuracies would indicate a better vector. This can be used as a way to compare different embeddings being used. It would be expected that embeddings that perform better with this evaluation would also perform better for use in the desired application. However for specific applications this can be tested to see if this evlaution method does correspond to the best performance.


```python
# Use GloVe evaluation methods to verify vectors
# perform a comparison betwen the regular length and
# normalized vectors
from evaluate import evaluate_vectors
print("---- normalized vector results: -----")
evaluate_vectors(W_norm, vocab, ivocab)

print("---- original vector results: ----")
evaluate_vectors(W, vocab, ivocab)
```

    ---- normalized vector results: -----
    capital-common-countries.txt:
    ACCURACY TOP1: 79.25% (401/506)
    capital-world.txt:
    ACCURACY TOP1: 68.48% (3098/4524)
    currency.txt:
    ACCURACY TOP1: 8.31% (72/866)
    city-in-state.txt:
    ACCURACY TOP1: 15.32% (378/2467)
    family.txt:
    ACCURACY TOP1: 68.97% (349/506)
    gram1-adjective-to-adverb.txt:
    ACCURACY TOP1: 15.22% (151/992)
    gram2-opposite.txt:
    ACCURACY TOP1: 9.48% (77/812)
    gram3-comparative.txt:
    ACCURACY TOP1: 51.80% (690/1332)
    gram4-superlative.txt:
    ACCURACY TOP1: 28.61% (321/1122)
    gram5-present-participle.txt:
    ACCURACY TOP1: 41.57% (439/1056)
    gram6-nationality-adjective.txt:
    ACCURACY TOP1: 85.99% (1375/1599)
    gram7-past-tense.txt:
    ACCURACY TOP1: 37.50% (585/1560)
    gram8-plural.txt:
    ACCURACY TOP1: 59.91% (798/1332)
    gram9-plural-verbs.txt:
    ACCURACY TOP1: 34.37% (299/870)
    Questions seen/total: 100.00% (19544/19544)
    Semantic accuracy: 48.46%  (4298/8869)
    Syntactic accuracy: 44.36%  (4735/10675)
    Total accuracy: 46.22%  (9033/19544)
    ---- original vector results: ----
    capital-common-countries.txt:
    ACCURACY TOP1: 58.50% (296/506)
    capital-world.txt:
    ACCURACY TOP1: 32.12% (1453/4524)
    currency.txt:
    ACCURACY TOP1: 8.08% (70/866)
    city-in-state.txt:
    ACCURACY TOP1: 6.53% (161/2467)
    family.txt:
    ACCURACY TOP1: 34.78% (176/506)
    gram1-adjective-to-adverb.txt:
    ACCURACY TOP1: 0.60% (6/992)
    gram2-opposite.txt:
    ACCURACY TOP1: 0.62% (5/812)
    gram3-comparative.txt:
    ACCURACY TOP1: 8.93% (119/1332)
    gram4-superlative.txt:
    ACCURACY TOP1: 2.85% (32/1122)
    gram5-present-participle.txt:
    ACCURACY TOP1: 7.58% (80/1056)
    gram6-nationality-adjective.txt:
    ACCURACY TOP1: 69.17% (1106/1599)
    gram7-past-tense.txt:
    ACCURACY TOP1: 8.27% (129/1560)
    gram8-plural.txt:
    ACCURACY TOP1: 26.35% (351/1332)
    gram9-plural-verbs.txt:
    ACCURACY TOP1: 1.49% (13/870)
    Questions seen/total: 100.00% (19544/19544)
    Semantic accuracy: 24.31%  (2156/8869)
    Syntactic accuracy: 17.25%  (1841/10675)
    Total accuracy: 20.45%  (3997/19544)


It seems the normalized vector results are performing better. The next thing to note is that the definition of the cosine distance is a dot product of vectors, normalized by the distance. This will be explained in further details in coming sections. 

For efficiency, the cosine similarities are often calulated without normalizing by the vector length. Therefore we will compare the word similarities using the normalized and full length vectors. In the results below, it can be seen that the results are similar though there are differences. Since this is not a quantitative method of evaluation it is not easy to say which word similarities are better, however since the technical definintion of cosine distance relies on normalized vectors it would be advisable to normalize the vectors, but also keep in mind that both ways could be tested to see the effects on the results for a given application. Perhaps in some cases preserving the length of the vectors would be desirable.


```python
# import distance measuring function from GloVe
from distance2 import distance
print("---- original vector results: -----")
distance(W, vocab, ivocab, 'king')
print("---- normalized vector results: -----")
distance(W_norm, vocab, ivocab, 'king')
```

    ---- original vector results: -----
    Word: king  Position in vocabulary: 691
    
                                   Word       Cosine distance
    
    ---------------------------------------------------------
    
                                emperor		5.061746
    
                                 throne		4.344266
    
                                    son		4.295550
    
                                   lord		4.223305
    
                                 prince		4.186335
    
                                     ii		4.087398
    
                                  queen		4.070626
    
                                dynasty		3.983273
    
                                kingdom		3.959915
    
                                  ruler		3.938324
    
    ---- normalized vector results: -----
    Word: king  Position in vocabulary: 691
    
                                   Word       Cosine distance
    
    ---------------------------------------------------------
    
                                 prince		0.823618
    
                                  queen		0.783904
    
                                     ii		0.774623
    
                                emperor		0.773625
    
                                    son		0.766719
    
                                  uncle		0.762715
    
                                kingdom		0.754216
    
                                 throne		0.753991
    
                                brother		0.749241
    
                                  ruler		0.743425
    


Next we will show how to get the distance between two words. This calculation will form the basis for document comparisons to come. To do this we will use our own implemented function using the built in python cosine distance measurement to assess the difference between two words, by extracting their embedding vector from the normalized matrix.


```python
def calc_cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)

# compare glove word embeddings
v1 = W_norm[vocab['king'], :]
v2 = W_norm[vocab['queen'], :]
sim = calc_cosine_similarity(v1, v2)
print(sim)
```

    0.783904300999


It is clear that the custom function for calculating cosine similarity is producing the same result as the GloVe calculation, so it will be safe to use this for other algorithms.

## Document Representations and Similarities

In this next section we will build upon the previous section to generate document representations from the word embeddings for each movie. We will also present alternative ways of representing movies, using the metadata, and using a more simple approach of representing the descriptions of the movies with the "Term Frequency - Inverse Document Frequency" (tfidf) approach. These three different methods of representing documents will be assessed on how well they are able to identify similar movies based on these text features.

### Document Embedding Vectors

This section will show how to represent documents with an embedding vector and then show how to use the movie embedding to get a list of similar movies.

#### Generating the Embedding Vectors

Next we represent each document as a vector based on the embeddings of the words it contains. This can be done by retreiving the word embeddings for each word in the list of tokens, and then taking the centroid of these vectors to represent the document as one vector. GloVe pretrained embeddings try to optimize storing the maximum representation of words in a much lower dimensional space. To do this, we have a function called movie_vector, that will create an embedding vector for a given movie, with the input as tokens of the movie description. In order to use this function it is required to first tokenize the movie description, then get the word vectors as shown previously.


```python
# returns normalized vector rep of the movie
# vector_dim as previously defined the dimension of one word vector
# this function relies on previously generated data vocabs and matrices

def movie_vector(W, movie_toks):
    vec_result = np.zeros(vector_dim)
    for idx, term in enumerate(movie_toks):
        if term in vocab:
            #print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :] 
        else:
            #print('Word: %s  Out of dictionary!\n' % term)
            continue
    
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    if d > 0:
        vec_norm = (vec_result.T / d).T        
    return vec_norm

# test this function to get a vector for one movie
# Note this is using the title2index but the same could be 
# done from any other id mapping to index
title = 'Mean Girls'
idx = title2index[title] # the mapping to matrix index
smd[smd.index == idx]
movie_toks = tokens[idx] # tokens list also based on this index
print("Mean Girls movie vector:")
print(movie_vector(W_norm, movie_toks))
```

    Mean Girls movie vector:
    [ -4.93672053e-02   1.52430687e-01  -2.05389903e-02   2.05246823e-02
       1.62483148e-01   1.13093878e-01  -1.49768446e-01  -5.41984829e-02
      -6.18856759e-02   1.22158519e-02  -8.34710386e-03   9.66496158e-02
      -1.02698110e-01   2.01688507e-02   1.08731190e-01  -1.69683162e-02
      -5.59966195e-02   7.14464739e-02  -6.86821838e-02  -2.02682901e-02
      -3.55507603e-02   1.17407977e-01   2.50736765e-04   1.26677554e-02
       2.54700024e-02  -4.29726582e-01  -4.74213006e-02   2.63490900e-02
      -4.61213273e-03  -1.72475568e-01   7.19597129e-01   5.77104464e-03
       2.60526024e-02  -7.10883941e-03   5.98140938e-02   1.86511793e-02
       3.62828345e-02  -4.47221975e-02   6.86866311e-02  -1.18390586e-01
      -9.36009285e-02   5.99338586e-02  -9.12574109e-03  -1.04849536e-01
       1.25914916e-01   8.00522571e-03  -2.91040108e-02  -1.89303337e-01
       2.73562960e-02   6.73045993e-02]


So now that we can get a vector representation of a movie description, we can use this to create a matrix of all the movies. This will be consructed with the following code, filling each row of the array with the movie vector for that movie.


```python
# create a matrix length of tokens (one set of tokens for each movie)
# the width of the matrix is the dimension of the embedding vectors.
movie_desc = np.zeros((len(tokens), vector_dim))

i = 0
# fill in the matrix row by row for each movie tokens
# the corresponding index of the arrays will be the same as titles
# and ids mapping to indexes
for movie_toks in tokens:
    # words not found in embedding index will be all-zeros.
    # use the normalized word embeddings W_norm
    movie_desc[i] = movie_vector(W_norm, movie_toks)
    i = i+1

# Check the result embedding matrix dimensions
print(movie_desc.shape)
```

    (9219, 50)


#### Assessing Movie Similarities

Now that we have a representation of movies as a vector, we can use this to compute similarity to other movies. This will be done with modified code from the GloVe github account.

The following function will take in a desired matrix of word embeddings W, the movie embeddings matrix M, the vocab mappings, and the tokens for a desired movie. To use this function, first the tokens for the desired move must be retrieved from the tokens list, access from the index, and the the function can first compute a movie representation, and then find similar movies from the matrix M.


```python
# try using distance fucntion on movies
# M is the embedding matrix of movies
def movie2movie_distance(W, M, vocab, ivocab, movie_toks):
    N= 10
    # this will generate a movie vector from the tokens
    for idx, term in enumerate(movie_toks):
        if term in vocab:
            #print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :] 
        else:
            #print('Word: %s  Out of dictionary!\n' % term)
            continue
    
    # normalize the movie vector
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    # get the distance between this movie and othe movies
    # this is the cosine distance, dot product of normalize vecs
    dist = np.dot(M, vec_norm.T)
    
    a = np.argsort(-dist)[:N]

    print("\n                               Word       Cosine distance\n")
    print("---------------------------------------------------------\n")
    for x in a:
        print("%35s\t\t%f\n" % (titles.iloc[x], dist[x]))
```


```python
# call this Function with previously generated matrices
# get movie tokens from the tokens list
title = 'Mean Girls'
idx = title2index[title] # the mapping to matrix index
movie_toks = tokens[idx] # tokens list also based on this index

movie2movie_distance(W_norm, movie_desc, vocab, ivocab, movie_toks)
```

    
                                   Word       Cosine distance
    
    ---------------------------------------------------------
    
                             Mean Girls		1.000000
    
                          Deadly Friend		0.954311
    
               'Neath the Arizona Skies		0.949922
    
                            Phantasm II		0.944674
    
                        Another 48 Hrs.		0.943287
    
             House II: The Second Story		0.942317
    
                           The Freshman		0.941801
    
          Hello Mary Lou: Prom Night II		0.940905
    
                      Feeling Minnesota		0.940621
    
                      Crimes of Passion		0.940600
    


For a sanity check we left mean girls as the top entry to confirm it is a cosine distance of 1 as would be expected. The other movies in the list are quite strange and it interesting to think about why these movies might seem similar based on content to Mean Girls. Let's inspect the results for another movie.


```python
# call this Function with previously generated matrices
# get movie tokens from the tokens list
title = 'The Godfather'
idx = title2index[title] # the mapping to matrix index
movie_toks = tokens[idx] # tokens list also based on this index

movie2movie_distance(W_norm, movie_desc, vocab, ivocab, movie_toks)
```

    
                                   Word       Cosine distance
    
    ---------------------------------------------------------
    
                          The Godfather		1.000000
    
                             The Family		0.963330
    
                                Sisters		0.958095
    
                              Max Payne		0.953557
    
                       Casa De Mi Padre		0.953418
    
                    The Legend of Zorro		0.952327
    
            Once Upon a Time in America		0.951490
    
                      Gangs of New York		0.950990
    
                          Little Odessa		0.949527
    
                        The Deer Hunter		0.949502
    


These results are actually promising, as it identified Gangs of New York as a similar movie, and also other movies that seem they would have similar content. Whether these movies would appeal to the same users interested in The Godfather though is not verifiable at this point, and will be investigated later on.

### Term Frequency Inverse Document Frequency Representation

Another way of representing documents, is with the term frequency inverse document frequency (tfidf). This does not require a representation of words, instead each word in the vocabulary across all documents are represented in a vocabulary vector. For each movie, the weights in the vocabulary vector are calculated as a count of the occurences of the word in that document, and inversely proportional to the occurence of that word in other documents, which will result in a measure of specificty and importance of that word for that document. The result of this is a very sparse, high dimensional vector for each document that spans the entire vocabulary. The tfidf of the movie descriptions will be implemented with a function from python, and it will also incorporate its own tokenization.


```python
# create the tfidf vectorizer, and output the matrix (sparse)
tf = []
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=5, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])

# shape will show # docs, vocab
print(tfidf_matrix.shape)

# type will show what kind of matrix results
print(type(tfidf_matrix))

# sample of features found
print(tf.get_feature_names()[1000:1005])
```

    (9219, 11569)
    <class 'scipy.sparse.csr.csr_matrix'>
    ['based experiences', 'based life', 'based novel', 'based play', 'based real']


#### Sparse Representation Methods

The resulting tfidf generated from the data is a sparce matrix as show above, and it has a vocabulary of 11569. Note that in this case, each document is represented by a vector of dimension 11569 in length vs 50 dimensions in length from the GloVe representation. Obviously this vector will be much more sparse, and therefore can make use of efficient python functions optimized for sparse matrices. In order to calculate the cosine distance between two movies using this representation, it is possible to use a built in python function that will simply compute the dot product between these two vectors. Note that the full cosine distance would rewuire to also normalize by vector length, however since this is an efficient function we will first assess results using this method.

We will use sklearn's linear_kernel instead of cosine_similarities and the output is a numpy array matrix of cosine similarities between each document.


```python
# This simple approach can only be applied on the sparse matrix
tfidf_sp_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(type(tfidf_sp_sim))
print(tfidf_sp_sim.shape)
```

    <class 'numpy.ndarray'>
    (9219, 9219)


Note the resulting matrix is a square matrix, with the where the metric  $dist(u=X[i],v=X[j])$  is computed and stored in entry  $ij$. This matrix can next be used to calculate the most similar movies. To do this we will make a function to generate recommendations from this matrix for a desired movie.


```python
# once given a simlarity matrix, this function will be the same everytime, change inputs
def get_recommendations(title, sim_matrix):
    idx = title2index[title]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #sim_scores = sorted(sim_scores, key=lambda x: 1-x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    scores = [score[1] for score in sim_scores]
    #scores = [1-score[1] for score in sim_scores]
    movie_indices = [i[0] for i in sim_scores]
    top_titles = titles.iloc[movie_indices]
    df = pd.DataFrame()
    df['titles'] = top_titles
    df['scores'] = scores
    return df
```


```python
# the results of the calculation on the sparse matrix
print("Sparse matrix calculation results:")
print(get_recommendations('The Godfather', tfidf_sp_sim).head())
```

    Sparse matrix calculation results:
                          titles    scores
    8494              The Family  0.251606
    994   The Godfather: Part II  0.244506
    4237      Johnny Dangerously  0.189206
    3536                    Made  0.182327
    29            Shanghai Triad  0.143431


It is interesting that this method also returned the movie "The Family" as its top hit, which is the same as the embedding method. However this method actually seems to be a bit better of a suggestion list, since it is recommending the sequel to the godfather, which definitely would be of interest to users who watched The Godfather.

#### Dense Calculation Methods

In order to compare to the embedding methods, we will change the tfidf to a regular (non-sparse) matrix representation, and test the document similarites using the same cosine function used previoulsy measuring the distance between the movie embedding vectors.

To do this first we will create the regular numpy matrix, and the perform the calculation of cosine similarity.


```python
# Since this is a sparse matrix, create a regular version too
# This will let the embedding matrices to come be compared 
# directly using the same functions
tfidf_np = np.asarray(tfidf_matrix.toarray())
print(tfidf_np.shape)
print(type(tfidf_np))
```

    (9219, 11569)
    <class 'numpy.ndarray'>



```python
# take the index of the movie in the tfidf matrix
# take in the tfidf matrix
# compute the distance between each movie and the desired movie
def movie2movie_tfidf(M, movie_index):
    N= 10
    vec_result = M[movie_index]
    
    # normalize the movie vector
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    # get the distance between this movie and other movies
    # this is the cosine distance, dot product of normalized vecs
    dist = np.dot(M, vec_norm.T)
    
    a = np.argsort(-dist)[:N]

    print("\n                               Word       Cosine distance\n")
    print("---------------------------------------------------------\n")
    for x in a:
        print("%35s\t\t%f\n" % (titles.iloc[x], dist[x]))
```


```python
# test this function on the movie index of The Godfather
title = 'Mean Girls'
idx = title2index[title] # the mapping to matrix index

movie2movie_tfidf(tfidf_np, idx)
```

    
                                   Word       Cosine distance
    
    ---------------------------------------------------------
    
                             Mean Girls		1.000000
    
                             Wild Child		0.174726
    
                          Pitch Perfect		0.158836
    
                              The Craft		0.156621
    
                            Latter Days		0.145837
    
                             The Clique		0.140224
    
                     Death at a Funeral		0.139270
    
                          Fallen Angels		0.136594
    
                          Doc Hollywood		0.135179
    
                      Mrs. Winterbourne		0.132972
    



```python
# test this function on the movie index of The Godfather
title = 'The Godfather'
idx = title2index[title] # the mapping to matrix index

movie2movie_tfidf(tfidf_np, idx)
```

    
                                   Word       Cosine distance
    
    ---------------------------------------------------------
    
                          The Godfather		1.000000
    
                             The Family		0.251606
    
                 The Godfather: Part II		0.244506
    
                     Johnny Dangerously		0.189206
    
                                   Made		0.182327
    
                         Shanghai Triad		0.143431
    
                      The Tillman Story		0.135222
    
          Elite Squad: The Enemy Within		0.131102
    
                                Thinner		0.126314
    
                               3 Ninjas		0.123779
    


This result is the same as calculated with the sparse kernel calculation. For sanity check The Godfather is included as the top result with a cosine distance of 1. 

It seems that the tfidf is providing better recommendations for both The Godfather and Mean Girls, than the embedding based on the suggested lists for these movies, however further investigation is required to see how this corresponds to the actual user preferences.

In the next section we will investigate one more approach to a content based recommender, incorporating the metadata of the movies. 

### Metadata Based Representation

The metadata recommender will take in metadata keywords and cast and crew metadata, previously processed into the "soup" column of the dataset. This will be a list of words, which can once again be tokenized and created into a vector. In this case, we will try the tfidf vector but also just a simple occurence based vector to represent each document. It is possible the tfidf might not make a lot of sense in this case, becuase odds are each word in the metadata will occure only once (ie. a list of keywords will list each word once), and so just indicating the occurence of each word may be enough. 

First we will proceed with the tfidf method shown previously.


```python
# This simple approach can only be applied on the sparse matrix
# create the tfidf vectorizer, and output the matrix (sparse)
meta_tf = []
meta_tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=5, stop_words='english')
meta_tfidf_matrix = meta_tf.fit_transform(smd['soup'])

# shape will show # docs, vocab
print(meta_tfidf_matrix.shape)

# type will show what kind of matrix results
print(type(meta_tfidf_matrix))

# sample of features found
print(meta_tf.get_feature_names()[100:105])

tfidf_meta = linear_kernel(meta_tfidf_matrix, meta_tfidf_matrix)
print(type(tfidf_meta))
print(tfidf_meta.shape)
```

    (9219, 7575)
    <class 'scipy.sparse.csr.csr_matrix'>
    ['adamsandler', 'adamshankman', 'adamshankman adamshankman', 'adamshankman adamshankman adamshankman', 'adamstorke']
    <class 'numpy.ndarray'>
    (9219, 9219)



```python
tfidf_meta_cos = cosine_similarity(meta_tfidf_matrix, meta_tfidf_matrix)
print(type(tfidf_meta_cos))
print(tfidf_meta_cos.shape)
```

    <class 'numpy.ndarray'>
    (9219, 9219)


Looking at the shape of this tfidf vector, it is much smaller than the description based tfidf matrix. Also, it seems the metadata token results look quite strange, but what this is doing is it is taking into account the cast and crew members, as these members may have an impact on the qualifty of the movie produced. The next step will be to get recommendations based on this recommendation and see how they compare. For this, we will use the same function as shown previously.


```python
print("The Godfather:")
print(get_recommendations('The Godfather', tfidf_meta).head())
print("Mean Girls:")
print(get_recommendations('Mean Girls', tfidf_meta).head())
```

    The Godfather:
                                 titles    scores
    3616  Tucker: The Man and His Dream  0.717023
    994          The Godfather: Part II  0.695543
    4518             One from the Heart  0.692377
    3300               Gardens of Stone  0.674118
    1346                  The Rainmaker  0.655029
    Mean Girls:
                         titles    scores
    3319        Head Over Heels  0.732575
    4763          Freaky Friday  0.663749
    7905  Mr. Popper's Penguins  0.663324
    6277       Just Like Heaven  0.644200
    1329       The House of Yes  0.641305


These results also look like quite reasonable suggestions, though they are a bit different than the description based findings. We can assess the results as well with direct count based vector.


```python
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])

print(count_matrix.shape)

# test two methods of assessing cosine similarity
meta_count_cos = cosine_similarity(count_matrix, count_matrix)
meta_count_kern = linear_kernel(count_matrix, count_matrix)
```

    (9219, 107377)



```python
# store the vocabulary from this model for later use in test
tfidf_vocab = count.vocabulary_
print(count_matrix.shape)
len(tfidf_vocab.keys())
```

    (9219, 107377)





    107377



In assessing the results we will use the built in cosine similarity function which normalizes results, as well as the same function as previously used which is the linear kernel to compute the dot product. The results are quite similar but it looks like the kernel similarity is actually producing better results, as it recommends both Godfather Part 2 and Part 3 very close to the top. This looks like the best set of recommendations so far. 


```python
print("------ Cosine similarity function: -------")
print(get_recommendations('The Godfather', meta_count_cos).head(10))
print("------ Linear Kernel function (same as previous): ------")
print(get_recommendations('The Godfather', meta_count_kern).head(10))
```

    ------ Cosine similarity function: -------
                                 titles    scores
    3616  Tucker: The Man and His Dream  0.477455
    994          The Godfather: Part II  0.456630
    1346                  The Rainmaker  0.431641
    3705                The Cotton Club  0.407661
    4518             One from the Heart  0.406297
    3300               Gardens of Stone  0.406269
    1602        The Godfather: Part III  0.375081
    2998               The Conversation  0.369835
    5867                    Rumble Fish  0.369835
    1992          Peggy Sue Got Married  0.366774
    ------ Linear Kernel function (same as previous): ------
                                 titles  scores
    994          The Godfather: Part II    21.0
    1346                  The Rainmaker    18.0
    1602        The Godfather: Part III    18.0
    3705                The Cotton Club    17.0
    981                  Apocalypse Now    16.0
    3300               Gardens of Stone    16.0
    1691                  The Outsiders    15.0
    2998               The Conversation    15.0
    3616  Tucker: The Man and His Dream    15.0
    4518             One from the Heart    15.0


## Representing User Profiles

Now that we have the ability to represent movies as vectors and compute similiarities between movies, we will extend the recommendations to take into account user preferences. For each user we will generate a profile based on the movies they previously liked. This will be done by using the previously generated movie vectors. Then, to provide recommendations, similarity measurements will be done using modified version of the recommendation functions to take in a user profile.

User profiles will be constructed based on each of the three previous types of document vectors.

### Embedding Method User Profiles

The embedding for a user profile can be created from the movie vectors they like in a similar way that the movie vectors were created from the words, by adding all the vectors and normalizing them, equivalent to taking the centroid of these vectors. This method will be applied in the following function which is a slight modification of the movie2movie function shown previously. However first it is required to calculate a vector representation of the user to pass into this function. This will be done with the user_vector function, which is similar to the movie_vector function.

The user vector will be created by taking in the matrix of previously computed movie embeddings, and the list of movies the user liked. The list of movies the user liked can be obtained from the ranking dataset for a given user id.


```python
# will generate an embedding for a user with the embedding matrix of movies M
# wil take in a list of movie names

def user_vector(M, movies):
    vec_result = np.zeros(M.shape[1])
    #print(vec_result.shape)
    for idx, movie in enumerate(movies):
        # indices is the mapping of movies titles to the index
        if movie in title2index:
            # this is the position of the movie in the array
            idm = title2index[movie]
            
            # hack solution to account for multiple movies of
            # same title --> should be replaced with tmdb id
            if type(idm) is not np.int64:
                idm = idm.iloc[0]
                
            if idx == 0:
                vec_result = np.copy(M[idm, :])
            else:
                vec_result += M[idm, :]
        else:
            print('Movie: %s  Out of dictionary!\n' % movie)
            continue
    
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    if d > 0:
        vec_norm = (vec_result.T / d).T        
    return vec_norm
    #return vec_result
```


```python
# Test this function with an example where a user 
# liked just The Godfather to compare to previous results
user_vec = user_vector(movie_desc, ['The Godfather'])
```

To confirm that this result is comparable to the results generated previously, we will create a function for calculating the user to movie distance, similar to the previously function for movie to movie distance. This function is shown below and it will be used to test the user vector of just The Godfather movie.


```python
# try using distance fucntion on movies
# M is the embedding matrix of movies
def user2movie_distance(M, vocab, ivocab, usr_prof):
    N= 10
    
    # normalize the movie vector
    vec_norm = np.zeros(usr_prof.shape)
    d = (np.sum(usr_prof ** 2,) ** (0.5))
    vec_norm = (usr_prof.T / d).T

    # get the distance between this movie and othe movies
    # this is the cosine distance, dot product of normalize vecs
    dist = np.dot(M, vec_norm.T)
    
    a = np.argsort(-dist)[:N]

    print("\n                               Word       Cosine distance\n")
    print("---------------------------------------------------------\n")
    for x in a:
        print("%35s\t\t%f\n" % (titles.iloc[x], dist[x]))
```


```python
# test the user_vec
user2movie_distance(movie_desc, vocab, ivocab, user_vec)
```

    
                                   Word       Cosine distance
    
    ---------------------------------------------------------
    
                          The Godfather		1.000000
    
                             The Family		0.963330
    
                                Sisters		0.958095
    
                              Max Payne		0.953557
    
                       Casa De Mi Padre		0.953418
    
                    The Legend of Zorro		0.952327
    
            Once Upon a Time in America		0.951490
    
                      Gangs of New York		0.950990
    
                          Little Odessa		0.949527
    
                        The Deer Hunter		0.949502
    


This function is therefore working, since it is generating the same values seen previously from direct movie to movie recommendations. Next we shall test on an actual user profile. To use this function, we first need to generate a list of movies that a user liked. This can be done with the following function.


```python
# function to get a user profile based on liked content
def user_liked_movies(userID, ratings):
    userRated = ratings[ratings['userId'] == userID]
    positiveRatings = userRated[ratings['rating'] > 3]
    likedMovies = positiveRatings['movieId']

    likedMovie_tmdb = []
    likedMovie_title = []
    for movieID in likedMovies:
        # get the associated tmdb id
        if movieID in movieid_map.index:
            #print("1. movieId -------------------")
            #print(movieID)
            tmdb_id = movieid_map.loc[movieID]['id']
            #print("tmdb_id")
            #print(tmdb_id)
            # store the tmdb ids
            likedMovie_tmdb.append(tmdb_id)
            # map the tmdb id to the movie title
            if type(tmdb_id) is not np.float:
                tmdb_id = tmdb_id.iloc[0]
            
            idx = id2index[tmdb_id]
            #print("idx")
            #print(idx)
            if type(idx) is not np.int64:
                idx = idx.iloc[0]
            
            #title = smd['title'][smd.index == idx]
            title = titles[idx]
            #print("title")
            #print(title)
            likedMovie_title.append(title)
    return likedMovie_title
```

Let us test this function for a random user and see what movies they liked. It iwl be printed out in the following code.


```python
usr_num = 300
userRated = ratings[ratings['userId'] == usr_num]
positiveRatings = userRated[ratings['rating'] > 3]
likedMovies = positiveRatings['movieId']

likedMoviesList = user_liked_movies(usr_num, ratings)
print(likedMoviesList)
```

    ['Twelve Monkeys', 'Braveheart', 'Apollo 13', 'Star Wars', 'Forrest Gump', 'Jurassic Park', 'Terminator 2: Judgment Day', 'The Silence of the Lambs', 'The Rock', 'That Thing You Do!', 'Raiders of the Lost Ark', 'Return of the Jedi', 'Indiana Jones and the Last Crusade', 'Die Hard 2', 'Mars Attacks!', 'The Fifth Element', 'Batman & Robin', 'Air Force One', 'The Game', 'Armageddon', 'Lethal Weapon 4', 'Lethal Weapon 2', 'The Mask of Zorro', 'Blade', 'Planet of the Apes', 'The Mummy', 'Star Wars: Episode I - The Phantom Menace', 'Austin Powers: The Spy Who Shagged Me', 'American Beauty', 'Sleepy Hollow', 'The Green Mile', 'Close Encounters of the Third Kind', 'Gladiator', 'Mission: Impossible II', 'The Patriot', 'Crouching Tiger, Hidden Dragon', 'Cast Away', 'Minority Report']


This is quite a diverse set of ratings, however lets see how the similarity measures would assess this to produce recommentations. TO do this, we will use the two functions needs to first generate the list of movies, and then next, create a user vector, this will be done in the following function. This user seems to overall like action, horror and scifi and histrocial movies, lets see if any of these are detected in the recommendations.


```python
def user_profile(userID, ratings):
    likedMovies = user_liked_movies(userID, ratings)
    user_vec = user_vector(movie_desc, likedMovies)
    return user_vec
```


```python
# Use the user vector function to generate a profile
prof11 = user_profile(usr_num, ratings)

# get top movies for a user
user2movie_distance(movie_desc, vocab, ivocab, prof11)
```

    
                                   Word       Cosine distance
    
    ---------------------------------------------------------
    
           Once Upon a Time in the West		0.983114
    
                             The Spirit		0.979363
    
                             The Jacket		0.977636
    
                        Nothing to Lose		0.977626
    
    Captain America: The Winter Soldier		0.977007
    
                                      9		0.975849
    
        Star Trek II: The Wrath of Khan		0.975212
    
                  Phantasm IV: Oblivion		0.975141
    
                              First Kid		0.974670
    
                             ParaNorman		0.974628
    


It does seem to have recommended some exiting action based movies for this user. IN particular it is a good sign that it recommended Star Trek, since the user liked Star Wars, this seems like it might be a good idea. Next we can assess the same method of generating recommendations from the metadata vectors.

### Metadata User Profiles

To generate the metadata based profiles, we will use the same method of first finding the list of movies the user liked, then creating a vector based on these movies. Because the tfidf and the regular count vector performed similarly, from this point for the metadata user profile analysis we will just onctinue with count based for simplicity.


```python
# will generate an embedding for a user with the count matrix
# will take in a list of movie names, and the count matrix

def user_vector_meta(M, movies):
    vec_result = np.zeros((1, M.shape[1])) # vector will be same length
    for idx, movie in enumerate(movies):
        # indices is the mapping of movies titles to the index
        if movie in title2index:
            # this is the position of the movie in the array
            idm = title2index[movie]
            # hack solution to account for multiple movies of
            # same title --> should be replaced with tmdb id
            if type(idm) is not np.int64:
                idm = idm.iloc[0]
                
            #if idx == 0:
            #    vec_result = np.copy(M[idm, :])                
            #else:
            vec_result += M[idm, :]
        else:
            print('Movie: %s  Out of dictionary!\n' % movie)
            continue
    
    #print(vec_result.shape)
    #print(vec_result[0])
    #vec_norm = np.zeros(vec_result.shape)
    #d = (np.sum(vec_result ** 2,) ** (0.5))
    #if d > 0:
    #    vec_norm = (vec_result.T / d).T        
    return vec_result

def user_profile_meta(userID, movie_matrix, ratings):
    likedMovies = user_liked_movies(userID, ratings)
    user_vec = user_vector_meta(movie_matrix, likedMovies)
    return user_vec
```


```python
# once given a simlarity matrix, this function will be the same everytime, change inputs
def get_user_recommendations(prof, count_matrix):
    user_meta_sims = cosine_similarity(count_matrix, prof)
    sim_scores = list(enumerate(user_meta_sims))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #sim_scores = sorted(sim_scores, key=lambda x: 1-x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    scores = [score[1] for score in sim_scores]
    #scores = [1-score[1] for score in sim_scores]
    movie_indices = [i[0] for i in sim_scores]
    top_titles = titles.iloc[movie_indices]
    df = pd.DataFrame()
    df['titles'] = top_titles
    df['scores'] = scores
    return df
```

First we can test this function on just one movie as the input, and see how the results compare to the movie similarity results. This will be done in the code below. Looking at the results, it is a good sign that it is recommending the Godfather part 2 near the top of the list, the same as shown previously. Next we have an adjusted function to get the user recommednations, that will take in the user profile vector and the count matrix, then calculate the similarities between that vector and each movie using the cosine similarity function, and then output the top results


```python
# Test this function with an example where a user 
# liked just The Godfather to compare to previous results
user_vec = user_vector_meta(count_matrix, ['The Godfather'])
get_user_recommendations(user_vec, count_matrix).head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>titles</th>
      <th>scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3616</th>
      <td>Tucker: The Man and His Dream</td>
      <td>[0.477455260559]</td>
    </tr>
    <tr>
      <th>994</th>
      <td>The Godfather: Part II</td>
      <td>[0.456629651137]</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>The Rainmaker</td>
      <td>[0.43164102394]</td>
    </tr>
    <tr>
      <th>3705</th>
      <td>The Cotton Club</td>
      <td>[0.407660967054]</td>
    </tr>
    <tr>
      <th>4518</th>
      <td>One from the Heart</td>
      <td>[0.406296733866]</td>
    </tr>
  </tbody>
</table>
</div>



We can test this function once again on user number 300 and compare the results to results we got previously.


```python
# Use the user vector function to generate a profile for user 300
prof11 = user_profile_meta(usr_num, count_matrix, ratings)

# get top movies for a user
get_user_recommendations(prof11, count_matrix).head(20)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>titles</th>
      <th>scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1251</th>
      <td>The Lost World: Jurassic Park</td>
      <td>[0.370625966193]</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>Armageddon</td>
      <td>[0.356943476739]</td>
    </tr>
    <tr>
      <th>5869</th>
      <td>Twilight Zone: The Movie</td>
      <td>[0.346949005222]</td>
    </tr>
    <tr>
      <th>1062</th>
      <td>Indiana Jones and the Last Crusade</td>
      <td>[0.344380646326]</td>
    </tr>
    <tr>
      <th>6232</th>
      <td>War of the Worlds</td>
      <td>[0.341673363712]</td>
    </tr>
    <tr>
      <th>972</th>
      <td>Raiders of the Lost Ark</td>
      <td>[0.340651711682]</td>
    </tr>
    <tr>
      <th>427</th>
      <td>Jurassic Park</td>
      <td>[0.33946901943]</td>
    </tr>
    <tr>
      <th>1241</th>
      <td>The Fifth Element</td>
      <td>[0.329456655949]</td>
    </tr>
    <tr>
      <th>232</th>
      <td>Star Wars</td>
      <td>[0.327395220779]</td>
    </tr>
    <tr>
      <th>2120</th>
      <td>Star Wars: Episode I - The Phantom Menace</td>
      <td>[0.326242953738]</td>
    </tr>
    <tr>
      <th>1498</th>
      <td>Lethal Weapon 4</td>
      <td>[0.324961021804]</td>
    </tr>
    <tr>
      <th>1580</th>
      <td>Lethal Weapon 2</td>
      <td>[0.320809011678]</td>
    </tr>
    <tr>
      <th>983</th>
      <td>Return of the Jedi</td>
      <td>[0.312446308709]</td>
    </tr>
    <tr>
      <th>2788</th>
      <td>Close Encounters of the Third Kind</td>
      <td>[0.31016389337]</td>
    </tr>
    <tr>
      <th>3674</th>
      <td>Planet of the Apes</td>
      <td>[0.303785575676]</td>
    </tr>
    <tr>
      <th>2043</th>
      <td>Planet of the Apes</td>
      <td>[0.302834432877]</td>
    </tr>
    <tr>
      <th>7024</th>
      <td>Indiana Jones and the Kingdom of the Crystal S...</td>
      <td>[0.300160146696]</td>
    </tr>
    <tr>
      <th>6242</th>
      <td>The Island</td>
      <td>[0.297540370567]</td>
    </tr>
    <tr>
      <th>1692</th>
      <td>Indiana Jones and the Temple of Doom</td>
      <td>[0.29349220226]</td>
    </tr>
    <tr>
      <th>522</th>
      <td>Terminator 2: Judgment Day</td>
      <td>[0.291468118107]</td>
    </tr>
  </tbody>
</table>
</div>



Recall the original movies liked by the user were:
['Braveheart', 'Apollo 13', 'Star Wars', 'Jurassic Park', 'The Silence of the Lambs', 'Die Hard 2', 'Mars Attacks!', 'Batman & Robin', 'Lethal Weapon 2', 'The Mask of Zorro', 'The Mummy']

According to this it actually recommended all the movies the user liked in the top ten. Extedning the list to see what else it recommends, these recommendations all look very reasonable given the previously liked items, and this technique seems to be much more effective than the word embeddings method.

## Collaborative Filtering

The next approach will be to provide user recommendations entirely based on user ratings for purposes of comparison. This technique is used very commonly and is one of the main approaches to building recommedner systems. It works entirely based on the the ratings different users gavre items and finds patterns in this data with matrix factorization to provide recommendations of movies users have not yet rated. 

To implement collaborative filtering on this data set, we will use the the data that was loaded and split previously in the data loading section, and use frunctions from the pythonn surprise library to run the collaborative filtering algorithm, "singular value decomposition". This will also be evaluated with a built in evaultaion method.


```python
svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])
```

    Evaluating RMSE, MAE of algorithm SVD.
    
    ------------
    Fold 1
    RMSE: 0.9059
    MAE:  0.6976
    ------------
    Fold 2
    RMSE: 0.8873
    MAE:  0.6835
    ------------
    Fold 3
    RMSE: 0.9031
    MAE:  0.6950
    ------------
    Fold 4
    RMSE: 0.8928
    MAE:  0.6896
    ------------
    Fold 5
    RMSE: 0.8921
    MAE:  0.6871
    ------------
    ------------
    Mean RMSE: 0.8963
    Mean MAE : 0.6906
    ------------
    ------------





    CaseInsensitiveDefaultDict(list,
                               {'mae': [0.69764777436615044,
                                 0.68353114042359508,
                                 0.69495540034901448,
                                 0.68957784720269488,
                                 0.68706908980686643],
                                'rmse': [0.90591356192194716,
                                 0.88734077023277758,
                                 0.90310642584720202,
                                 0.89284567424961658,
                                 0.89208394615155329]})



The evaluation method is showing the Root Mean Squared error for predictions, which in this case for each fold of the training data is around 0.89, which is a good result. Next we can build a training dataset and use svd to train a model.


```python
trainset = data.build_full_trainset()
svd.train(trainset)
```

Let's assess the results for user 300, to see how the model will predict a movie that was produced in the top list of movies with the metadata content recommender. The movie chosen is "The Green Lantern" shown in the previous chart to have movie id 7903. The user was predicted to have rated this movie with a 3.9. This is a good rating, so this corresponds nicely with giving this movie as a recommendation based on its content.


```python
# predict results for user 300 with movie id 7903
svd.predict(300, 7903, 3)
```




    Prediction(uid=300, iid=7903, r_ui=3, est=3.8621103104241525, details={'was_impossible': False})



Next we can try this again with one movie that was recommended with the content based recommender. Lets choose the Captain America movie, and see how this method predicts the user would have rated it. As shown in the following code, the id for this movie is: 100402.


```python
smd[smd['title']=="Captain America: The Winter Soldier"]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>level_0</th>
      <th>index</th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>...</th>
      <th>vote_count</th>
      <th>year</th>
      <th>cast</th>
      <th>crew</th>
      <th>keywords</th>
      <th>description</th>
      <th>cast_size</th>
      <th>crew_size</th>
      <th>director</th>
      <th>soup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8626</th>
      <td>8626</td>
      <td>23249</td>
      <td>False</td>
      <td>{'id': 131295, 'name': 'Captain America Collec...</td>
      <td>170000000</td>
      <td>[Action, Adventure, Science Fiction]</td>
      <td>http://www.captainamericathewintersoldiermovie...</td>
      <td>100402</td>
      <td>tt1843866</td>
      <td>en</td>
      <td>...</td>
      <td>5881.0</td>
      <td>2014</td>
      <td>[chrisevans, samuell.jackson, scarlettjohansson]</td>
      <td>[{'gender': 1, 'job': 'Casting', 'department':...</td>
      <td>[washingtond.c., futur, shield, marvelcom, sup...</td>
      <td>After the cataclysmic events in New York with ...</td>
      <td>70</td>
      <td>41</td>
      <td>[anthonyrusso, anthonyrusso, anthonyrusso]</td>
      <td>washingtond.c. futur shield marvelcom superher...</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 35 columns</p>
</div>




```python
# predict results for user 300 with movie id 100402
svd.predict(300, 100402, 3)
```




    Prediction(uid=300, iid=100402, r_ui=3, est=3.8621103104241525, details={'was_impossible': False})



This movie was once again predicted to be rated quite highly, with a rating of 3.9 again, for this user so this is a good sign that even the embedding method can provide some decent recommendations.

## Hybrid Method

Now we will assess a way to incorporate both the content based recommendations and the user ratings in predictions. There are many ways this could be applied. Often recommender systems incorporate many different features. However for the purposes of this evaulation, we will focus on one method of combining these two approaches: providing a user recommendations based on a specific movie that have selected, and their preferences. This means that the movie similarities will be used in identifying a list of movies similar to one a user selected, and then the collaborative filtering predictions will be used to select movies the user would like the most out of that list. This can be seen in the following code for a hybrid recommender.


```python
# This will take in the user id and the seelcted title
def hybrid(userId, title):
    idx = title2index[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']
    
    # find similar movies to the given title
    sim_scores = list(enumerate(meta_count_cos[int(idx)]))
    #print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    #print(sim_scores)
    
    movie_indices = [i[0] for i in sim_scores]
    
    # use the svd predictions to get the top predictions
    #movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies = smd.iloc[movie_indices][['title', 'id']]
    #print(movies)
    
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, tmdb_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)
```


```python
# test this function for user 300, with movie "The Godfather"
hybrid(300, 'The Godfather')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>id</th>
      <th>est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>284</th>
      <td>The Shawshank Redemption</td>
      <td>278</td>
      <td>4.891969</td>
    </tr>
    <tr>
      <th>2998</th>
      <td>The Conversation</td>
      <td>592</td>
      <td>4.411174</td>
    </tr>
    <tr>
      <th>994</th>
      <td>The Godfather: Part II</td>
      <td>240</td>
      <td>4.382663</td>
    </tr>
    <tr>
      <th>2808</th>
      <td>Midnight Express</td>
      <td>11327</td>
      <td>4.381289</td>
    </tr>
    <tr>
      <th>986</th>
      <td>GoodFellas</td>
      <td>769</td>
      <td>4.375181</td>
    </tr>
    <tr>
      <th>981</th>
      <td>Apocalypse Now</td>
      <td>28</td>
      <td>4.122711</td>
    </tr>
    <tr>
      <th>1765</th>
      <td>The Paradine Case</td>
      <td>31667</td>
      <td>3.991149</td>
    </tr>
    <tr>
      <th>146</th>
      <td>Feast of July</td>
      <td>259209</td>
      <td>3.966451</td>
    </tr>
    <tr>
      <th>3300</th>
      <td>Gardens of Stone</td>
      <td>28368</td>
      <td>3.869151</td>
    </tr>
    <tr>
      <th>868</th>
      <td>Looking for Richard</td>
      <td>42314</td>
      <td>3.852843</td>
    </tr>
  </tbody>
</table>
</div>



So this list of recommendations is very different from the list seen previoulsy for this user, becaus they selected they wanted movies similar to The Godfather. The result of this is alist consisting of movies similar to The Godfather, ranked based on the predicted rating that user would have given it. It can be seen the top movies in this list are predicted to be rated wuite highly by the user, probably because these are overall well like popular movies, and also seem to fit within the genre of movies this user previously like. Now lets see how this method handles predictions for this user based on the selected movie "Mean Girls, which is a bit different in style of genre to the types of movies this user previoulsy liked.


```python
# test this function for user 300, with movie "The Godfather"
hybrid(300, 'Mean Girls')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>id</th>
      <th>est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1547</th>
      <td>The Breakfast Club</td>
      <td>2108</td>
      <td>4.134750</td>
    </tr>
    <tr>
      <th>7332</th>
      <td>Ghosts of Girlfriends Past</td>
      <td>12556</td>
      <td>4.033310</td>
    </tr>
    <tr>
      <th>6959</th>
      <td>The Spiderwick Chronicles</td>
      <td>8204</td>
      <td>4.001881</td>
    </tr>
    <tr>
      <th>7377</th>
      <td>I Love You, Beth Cooper</td>
      <td>19840</td>
      <td>3.975022</td>
    </tr>
    <tr>
      <th>5163</th>
      <td>Just One of the Guys</td>
      <td>24548</td>
      <td>3.946458</td>
    </tr>
    <tr>
      <th>390</th>
      <td>Dazed and Confused</td>
      <td>9571</td>
      <td>3.916099</td>
    </tr>
    <tr>
      <th>7084</th>
      <td>The Sisterhood of the Traveling Pants 2</td>
      <td>10188</td>
      <td>3.910710</td>
    </tr>
    <tr>
      <th>7436</th>
      <td>Reckless</td>
      <td>38702</td>
      <td>3.902787</td>
    </tr>
    <tr>
      <th>5092</th>
      <td>Lord Love a Duck</td>
      <td>52867</td>
      <td>3.900065</td>
    </tr>
    <tr>
      <th>6698</th>
      <td>It's a Boy Girl Thing</td>
      <td>37725</td>
      <td>3.898058</td>
    </tr>
  </tbody>
</table>
</div>



Note that the top rated movies predicted in this list are lower than the previous list, where the top movie here predicts the user would rate it at 4.3 vs 4.9 in the list previously for Shawshank Redemption. It is clear that the recommendation at the top of the list would have been suggested because it is a classic well liked movie within this genre. Other movie suggested, such as the Spiderwick Chronicles, as more action based teen targeted movies, so it is still picking up some of the user's preference towards more action style movies. It seems like a reasonable list given the inputs. 

## Evaluation of Results

Next we will try to get some quantitative measure of success of the different recommendation approaches. To do this we will assess the precision, recall, and F score. We will focus on the best recommender prodcued through assessment of the heuristic evaluations of movie lists shown in the previous section. The best recommender was the tfidf metadata recommender, and so now we will evaulate it with some standard metrics used in the literature for comparison to previous research.

To calculate the precision, we need to calculate the number of items in the top ten that were relevant to the user. Relevant here means items the user had positively rated. To do this we need to access the movieIds of the recommended movies, and look up in the ratings dataset the rating that the user gave that movie.


```python
# get list of recommended movies for user 300
recommendations = get_user_recommendations(prof11, count_matrix).head(10)
print(recommendations)

# Get a list of the tmbd ids from the recommended movies
rec_tmdb = ids[recommendations.index]
# get a list of the movie Ids from the list
rec_movieId = tmdb_map.loc[rec_tmdb]['movieId']
```

                                             titles            scores
    1251              The Lost World: Jurassic Park  [0.370625966193]
    1497                                 Armageddon  [0.356943476739]
    5869                   Twilight Zone: The Movie  [0.346949005222]
    1062         Indiana Jones and the Last Crusade  [0.344380646326]
    6232                          War of the Worlds  [0.341673363712]
    972                     Raiders of the Lost Ark  [0.340651711682]
    427                               Jurassic Park   [0.33946901943]
    1241                          The Fifth Element  [0.329456655949]
    232                                   Star Wars  [0.327395220779]
    2120  Star Wars: Episode I - The Phantom Menace  [0.326242953738]



```python
# Use the recommended movie Ids to look up ratings
user_rated_movies = ratings[ratings['userId'] == 300]
user_positive_movies = user_rated_movies[user_rated_movies['rating'] > 3]

# list of movie Ids:
positive_movieIds = [mid for mid in user_positive_movies['movieId']]
```

Next, to test the results of the model, we will split the data into a train and test set. This will assess the effectiveness of the model on unseen data. This is done in the following code using a random sample of ratings from the dataframe.


```python
# try splitting the ratings dataset into test and train
ratings_train= ratings.sample(frac=0.8,random_state=200)
ratings_test= ratings.drop(ratings_train.index)

```

Next we will use the method shown previsouly to access the user ratings from the train and test set, to create teh user profile from the training set, and then produce rankings from the test set. The tricky part to note in this implementation, is that the count matrix generated from the tfidf vectorizer needs to be created just out of data in the test set. In order to do this, the tfidf vectorizer is once again applied. The important thing to note is that it is taking in the full vocabulary previously generated, to result in a matrix of the same dimensions, to be able to get distance measure from teh user profile vector.

After this is done, the recommendations are found as usual from this matrix, and the number of relevant items produced in the recommendations is counted up.


```python
# function to calculate the number of relevant items
def user_rel_items(user_id, ratings_train, ratings_test):
    # need to generate user profile
    profile = user_profile_meta(user_id, count_matrix, ratings_train)
    
    # get out the movie ids
    movieIds_test = [mid for mid in ratings_test['movieId']]

    # get the corresponding tmdb ids, 'id' in smd
    tmdb_test = movieid_map.loc[movieIds_test]['id']
    tmdb_test_list = [tmdb for tmdb in tmdb_test]
    smd_test_idx = smd['id'].isin(tmdb_test_list)

    # try to access smd at same indice of test
    smd_test = smd[smd_test_idx]
    
    # need to only apply to indecies of smd in test
    count_test = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english', vocabulary=tfidf_vocab)
    count_matrix_test = count_test.fit_transform(smd_test['soup'])
    
    # need to get recommendations from user profile
    recommendations = get_user_recommendations(profile, count_matrix_test).head(10)
    
    # get the ids of the movies:
    # Get a list of the tmbd ids from the recommended movies
    rec_tmdb = ids[recommendations.index]
    # get a list of the movie Ids from the list
    rec_movieId = tmdb_map.loc[rec_tmdb]['movieId']
    
    # Get the list of movies the user rated positively in the test set
    user_rated_movies = ratings_test[ratings_test['userId'] == user_id]
    user_positive_movies = user_rated_movies[user_rated_movies['rating'] > 3]
    positive_movieIds = [mid for mid in user_positive_movies['movieId']]
    
    # Get the positive rated movies that were recommended:
    success_count = 0
    for movieId in rec_movieId:
        if movieId in positive_movieIds:
            success_count = success_count +1
    
    return success_count

# test the function for user 300
user_rel_items(300, ratings_train, ratings_test)    
```




    0



In the next sections of code, the precision recall and F measure functions will be defined. Also, there is a helper function for the recall that will return the total number of positive ratings in the test set for a user.


```python
# Next we can use this result to calculate precision, recall and F score
# we are keeping the top k, as 10. This function will be used 
# to calculate the overall precision across many users

def precision(tot_rel, num_users):
    k = 10
    prec = tot_rel/(num_users*k)
    return prec

prec300 = precision(user_rel_items(300, ratings_train, ratings_test), 1)

# this function will return the total number of positive ratings for a user
# this will be needed to calculate the recall

def user_pos_tot(user_id, ratings_test):
    # Get the list of movies the user rated positively
    user_rated_movies = ratings_test[ratings_test['userId'] == user_id]
    user_positive_movies = user_rated_movies[user_rated_movies['rating'] > 3]
    positive_movieIds = [mid for mid in user_positive_movies['movieId']]
    return len(positive_movieIds)

print("total positive ratings for user 300")
print(user_pos_tot(300, ratings_test))

# Recall is calculated as relevant found in top k/total relevant
# To use this function, first calculate the total relvant
# results, then calculate the total possible positive ratings

def recall(tot_rel, tot_pos):
    
    
    if tot_pos == 0:
        rec = 1
    else:
        rec = tot_rel/tot_pos
        
    return rec

rec300 = recall(user_rel_items(300, ratings_train, ratings_test), user_pos_tot(300, ratings_test))
```

    total positive ratings for user 300
    10


Next we can use the recall and precision to compute the F score. The result is calculated for user 300.


```python
def F_score(prec, rec):
    if prec+rec >0:
        F1 = 2*prec*rec/(prec+rec)
    else:
        F1 = 0
    return F1

F_score(prec300, rec300)
```




    0



Clearly this method does not perform well on unseen data. To be thorough we will assess the precision and recall for all users, and then once again compute the F1 measure.  In the next section of code we will extract all the user ids in the rating test data.


```python
# use these functions to calculate the scores across all users
# first get a list of all user ids in the ratings
all_users = [uid for uid in ratings_test['userId']]
all_users = set(all_users)
num_users = len(all_users)
print(len(all_users))
```

    671


Now to calculate the total recall and precision across all users, we will count up the total number of predictions in the top ten that were relevant predictions, which is done in the following code.


```python
rel_items = 0
tot_pos = 0
for user_id in all_users:
    rel_items = rel_items + user_rel_items(user_id, ratings_train, ratings_test)
    tot_pos = tot_pos + user_pos_tot(user_id, ratings_test)
    #print(rel_items, user_id)
    
```

Now that we have the total number of possible positive ratings per user, and the total number of relevant results returned to each user, we can use this to calculate overall precision and recall scores. This precision and recall scores can then be used to calculate the overall F measure across all users.


```python
p = precision(rel_items, num_users)
r = recall(rel_items, tot_pos)
F1 = F_score(p, r)
print("precision: ", p)
print("recall: ", r)
print("F1: ", F1)
```

    precision:  0.005216095380029807
    recall:  0.0028212155408673225
    F1:  0.003661853944339821


These numbers are much lower than seen in the literature. This is quite extreme, and reasons for this may be due to the method of calculating the evaluation scores. Since many users had ratings for movies within a certain rainge of similarity, perhaps instead of doing only the top ten, it would make more sense to see if the relevant movies could be found within a certain distance of similarity measure as the cutoff.

To analyze thse results a bit more, we will produce scores in the hypothetical situation that in the recall the maximum number of relevant items were returned in the top ten. i.e for each user the top ten was all completely correct. In this case we can see an upper bound for what the maxium highest recall score could be, and also see how this impacts the maximum possible F score possible at this level of k (k is the number of items in the reccomendations list).


```python
max_rel = 10*num_users

p = precision(rel_items, num_users)
r = recall(max_rel, tot_pos)
F1 = F_score(p, r)
print("precision: ", p)
print("maximum possible recall: ", r)
print("F1 with max recall: ", F1)
```

    precision:  0.005216095380029807
    maximum possible recall:  0.5408673222634209
    F1 with max recall:  0.010332544258684496


Note that with this perspective in mind of what the maximum acheivable recall score could be, this puts the results into a bit better perspective. 

Next, we will try one more approach of evaluating with this method. Instead of calculating overall relevant predictions, we will calculate the precision and recall individually for each user, and then take the average of this to produce final scores. These can then be used to calculate the overall average F measure.


```python
# calculate average precision and recalls
rec = 0
prec = 0
for user_id in all_users:
    rel_items = user_rel_items(user_id, ratings_train, ratings_test)
    tot_pos = user_pos_tot(user_id, ratings_test)
    prec = prec + precision(rel_items, 1)
    rec = rec + recall(rel_items, tot_pos)
    
avg_prec = prec/num_users
avg_rec = rec/num_users
F1_avg = F_score(avg_prec, avg_rec)
print("avergae precision: ", avg_prec)
print("avergae recall: ", avg_rec)
print("average F1: ", F1_avg)
```

    avergae precision:  0.005216095380029809
    avergae recall:  0.0262367939222
    average F1:  0.008702133419245498


Note that this way of evaluatiing results could offset some user that had rated many movies and therefore resulted in extremely low recalls, for example if they had one prediction in the top k, out of 1000 possible correct answers. Most users only rated on average 20 movies, so adding this large numbers of ratings to the sum for total positives could skew the results, but will have less of an effect when it is just effect the result of one user and then being averaged out.

Finally we can repeat the same procedure shown previously to see in this case how the maximium possible results would be. This will be shown in the following code.


```python
# calculate average precision and recalls
rec_max = 0
for user_id in all_users:
    tot_pos = user_pos_tot(user_id, ratings_test)
    # testing every recall assuming 10/10 relevant items
    # or all the positive items the user rated:
    if 10 < tot_pos:
        rec_max = rec_max + recall(10, tot_pos)
    else:
        rec_max= rec_max + recall(tot_pos, tot_pos)
    
avg_rec_max = rec_max/num_users
F1_avg = F_score(avg_prec, avg_rec_max)
print("Average max recall: ", avg_rec_max)
print("Average F1 with max recall: ", F1_avg)
```

Note that with the method of calculating the score that max recall is quite high. However the overall highest possible F1 measure is still similar as before within the same order of magnitude. This means that it is probably an effect of the dataset and these evaluation techniques that is contributing to the low results. While these scores could be used to compare model to model, it would be worthwhile to investigate further approaches to measure success of content based recommenders for future research.

### Modified F measure Calculation

In order to compare to previous sstudies, a modified version of the F measure was also calculuated. This modified function shown in the code below evalutes the top k scores only out of items each user rated. Therefore instead of in the testing having to predict out of all items in the test set, the predictions are only done out of the items the user rated, so there will be much fewer items to assess. On average users only rated 20 items so it makes sense that the results will be much higher when calculated this way. The same method as before was applied and the results are shown below. 


```python
def tot_rel(user_id, movie_matrix, ratings_train, ratings_test):
    profile = user_profile_meta(user_id, movie_matrix, ratings_train)

    # Get the list of movies the user rated positively in the test set
    user_rated_movies = ratings_test[ratings_test['userId'] == user_id]

    # get out the test movie ids
    movieIds_test = [mid for mid in user_rated_movies['movieId']]

    # get the corresponding tmdb ids, 'id' in smd to user's test movies
    tmdb_test = movieid_map.loc[movieIds_test]['id']
    tmdb_test_list = [tmdb for tmdb in tmdb_test]
    smd_test_idx = smd['id'].isin(tmdb_test_list)
    smd_test = smd[smd_test_idx]
    # will be number of items in test the user rated
    if len(movieIds_test)==0:
        print("no items in test")

    
    user_meta_sims = cosine_similarity(movie_matrix, profile)
    sim_scores = list(enumerate(user_meta_sims))
    user_rated_scores = []
    for idx in smd_test.index:
        #print(sim_scores[idx])
        user_rated_scores.append(sim_scores[idx])

    user_rated_scores_sort = sorted(user_rated_scores, key=lambda x: x[1], reverse=True)
    scores = [score[1] for score in user_rated_scores_sort]
    movie_indices = [i[0] for i in user_rated_scores_sort]

    # get the ids of the movies:
    # Get a list of the tmbd ids from the recommended movies
    rec_tmdb = ids[movie_indices]
    # get a list of the movie Ids from the list
    rec_movieId = tmdb_map.loc[rec_tmdb]['movieId']

    # Get the list of movies the user rated positively in the test set
    user_rated_movies = ratings_test[ratings_test['userId'] == user_id]
    user_positive_movies = user_rated_movies[user_rated_movies['rating'] > 3]
    positive_movieIds = [mid for mid in user_positive_movies['movieId']]

    success_count = 0
    topk = 10
    for movieId in rec_movieId:
        if topk == 0:
            #print("top 10 reached")
            break
        topk= topk-1
        if movieId in positive_movieIds:
            success_count = success_count +1
    return success_count
```


```python
user_id = 73
tot_rel(user_id, count_matrix, ratings_train, ratings_test)
```




    5




```python
# calculate average precision and recalls
rec = 0
prec = 0
rel = 0
for user_id in all_users:
    rel_items = tot_rel(user_id, count_matrix, ratings_train, ratings_test)
    rel = rel+rel_items
    #print(user_id, rel)
    tot_pos = user_pos_tot(user_id, ratings_test)
    prec = prec + precision(rel_items, 1)
    rec = rec + recall(rel_items, tot_pos)
    
```


```python
avg_prec = prec/num_users
avg_rec = rec/num_users
F1_avg = F_score(avg_prec, avg_rec)
print("avergae precision: ", avg_prec)
print("avergae recall: ", avg_rec)
print("average F1: ", F1_avg)
```

    avergae precision:  0.5709388971684057
    avergae recall:  0.6562348688608758
    average F1:  0.6106225910014146



```python
user_id = 73
tot_rel(user_id, meta_tfidf_matrix, ratings_train, ratings_test)
```




    5




```python
# calculate average precision and recalls
rec = 0
prec = 0
rel = 0
for user_id in all_users:
    rel_items = tot_rel(user_id, meta_tfidf_matrix, ratings_train, ratings_test)
    rel = rel+rel_items
    #print(user_id, rel)
    tot_pos = user_pos_tot(user_id, ratings_test)
    prec = prec + precision(rel_items, 1)
    rec = rec + recall(rel_items, tot_pos)
    
```


```python
avg_prec = prec/num_users
avg_rec = rec/num_users
F1_avg = F_score(avg_prec, avg_rec)
print("avergae precision: ", avg_prec)
print("avergae recall: ", avg_rec)
print("average F1: ", F1_avg)
```

    avergae precision:  0.5839046199701937
    avergae recall:  0.6591773210468267
    average F1:  0.6192619656656209


These results are much more comparable to the results in the reference study research paper! Actually, comparing the top 10 F measure values, these values computed using the tfidf actually outperform all the methods used in the past study. It must be taken into account however that the dimensions of these tfidf vectors for document representation are around 7000, vs the length 300 and 500 vectors used in the study. However, it should be noted that the study did not find significant improvements with the larger vector sizees (ie. 500 vs 300). Actually counter-intuitively the 300 sized document vecotr actually often outperformed the 500 dimension vector. Further investigation is require to assess the impact the length of the document representation vector has on the results.

## Conclusions

To fully evaulate recommenders it would be advisable to consider additional method of evaluation. The results of testing different recommenders showed that the lists of movies produced did heuristically seem like valid lists of recommendations. The best method shown was the metadata tdidf recommender. This recommender was then evaluated quantitatively, were similar but slightly higher than the values seen in the reference study in the literature. 
