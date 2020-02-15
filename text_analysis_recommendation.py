
import pandas as pd
from rake_nltk import Rake
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df_jokes = pd.read_csv("JokeText.csv")
print(df_jokes.head())
print(df_jokes.shape)

all_keywords=[]
for index, row in df_jokes.iterrows():
    joke = row['JokeText']
    joke = re.sub(r'[^\w\s]', '', joke)

    r = Rake()
    r.extract_keywords_from_text(joke)
    # getting the dictionary whith key words as keys and their scores as values
    key_words_dict_scores = r.get_word_degrees()

    all_keywords.append(list(key_words_dict_scores.keys()))

df_jokes['KeyWords'] = np.array(all_keywords)
# dropping the Joke text column
# df_jokes.drop(columns=['JokeText'], inplace=True)
# print(df_jokes.columns)
# print(df_jokes['KeyWords'].head(20))
df_jokes['KeyWords']=df_jokes['KeyWords'].str.join(" ")
# " ".join(my_list)
# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df_jokes['KeyWords'])
print(count_matrix.shape)
# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim)

def recommendations(JokeId, cosine_sim=cosine_sim):
    recommended_jokes = []

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[JokeId]).sort_values(ascending=False)

    top_10_indexes = list(score_series.iloc[1:11].index)

    for i in top_10_indexes:
        recommended_jokes.append(list(df_jokes.index)[i])

    return recommended_jokes

print(df_jokes['JokeText'].iloc[[2]])
print("************************")
for id in recommendations(2)[0:2]:
    print(df_jokes['JokeText'].iloc[[str(id)]])

def recommendations_user(UserId, number, df_rating):
    recommended_jokes = []

    top_n_jokes = pd.Series(df_rating['User'+str(UserId)]).sort_values(ascending=False)[0:number]
    print(top_n_jokes)
    for i in top_n_jokes.index:
        joke = recommendations(i)[0]
        if joke not in recommended_jokes:
            recommended_jokes.append(recommendations(i)[0])


    return recommended_jokes


# JokeNumber = 2
# print(df_jokes['JokeText'].iloc[[JokeNumber]])
# print("************************")
# for id in recommendations(JokeNumber)[0:2]:
#     print(df_jokes['JokeText'].iloc[[str(id)]])

# na osnovu 15 najbolje ocenjenih sala
num_jokes = 15
UserNumber = 3
for id in recommendations_user(UserNumber, num_jokes, df_rating):
    print(df_jokes['JokeText'].iloc[[str(id)]])
