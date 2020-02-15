import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_jokes = pd.read_csv("JokeText.csv")
print(df_jokes.head())
print(df_jokes.shape)

df_rating1 = pd.read_csv("UserRatings1.csv")
print(df_rating1.shape)
df_rating = df_rating1
res = df_rating.to_dict('index')

import math
lst=[]
for k, v in res.items():
    for a, b in v.items():
        if not math.isnan(b):
            if a!='JokeId':
                lst.append((k, a, b))
print(lst[0:10])
print(len(lst))
user_item_rating = pd.DataFrame(list(lst))
user_item_rating.columns = ['JokeId', 'UserId', 'Rating']
user_item_rating = user_item_rating[['UserId', 'JokeId', 'Rating']]
user_item_rating['UserId'] = [int(x[4:]) for x in user_item_rating['UserId']]
# df_new['UserId'] = df_new['UserId'].apply(lambda x: int(x[4:]))
print(user_item_rating.head(5))
print("******")
Strategy=3
if Strategy==1:
    user_item_rating = user_item_rating.head(800000)
elif Strategy==2:
    #  10000 korisnika koji su ocenili najvise sala
    users = user_item_rating.groupby('UserId')['Rating'].count().reset_index().sort_values('Rating', ascending=True)[-10000:]
    print(users.UserId.values)
    user_item_rating = user_item_rating[user_item_rating.UserId.isin(users.UserId.values)]
else:
    #  10000 korisnika koji su ocenili najmanje sala
    users = user_item_rating.groupby('UserId')['Rating'].count().reset_index().sort_values('Rating', ascending=True)[0:10000]
    print(users.UserId.values)
    user_item_rating = user_item_rating[user_item_rating.UserId.isin(users.UserId.values)]
print(user_item_rating.shape)

users_num = len(df_rating.columns)-1
jokes_num = df_jokes.JokeId.count()

print(f'No. of users: {users_num}')
print(f'No. of jokes: {jokes_num}')


rtg = user_item_rating.Rating.value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.rcParams.update({'font.size': 15}) # Set larger plot font size
plt.bar(rtg.index, rtg.values, width=0.1)
plt.xlabel('Rating')
plt.ylabel('counts')
plt.xlim(-10,10)
# plt.show()


from sklearn import model_selection
train_data, test_data = model_selection.train_test_split(user_item_rating, test_size=0.20)

print(f'Training set size: {len(train_data)}')
print(f'Testing set size: {len(test_data)}')
print(f'Test set is {(len(test_data)/(len(train_data)+len(test_data))*100):.0f}% of the full dataset.')

### TRAINING SET
# Get int mapping for user_id
u_unique_train = train_data.UserId.unique()  # create a 'set' (i.e. all unique) list of vals
train_data_user2idx = {o:i for i, o in enumerate(u_unique_train)}
# Get int mapping for unique_joke_id
b_unique_train = train_data.JokeId.unique()  # create a 'set' (i.e. all unique) list of vals
train_data_book2idx = {o:i for i, o in enumerate(b_unique_train)}

### TESTING SET
# Get int mapping for user_id
u_unique_test = test_data.UserId.unique()  # create a 'set' (i.e. all unique) list of vals
test_data_user2idx = {o:i for i, o in enumerate(u_unique_test)}
# Get int mapping for unique_joke_id
b_unique_test = test_data.JokeId.unique()  # create a 'set' (i.e. all unique) list of vals
test_data_book2idx = {o:i for i, o in enumerate(b_unique_test)}

### TRAINING SET
train_data['User_unique'] = train_data['UserId'].map(train_data_user2idx)
train_data['Joke_unique'] = train_data['JokeId'].map(train_data_book2idx)

### TESTING SET
test_data['User_unique'] = test_data['UserId'].map(test_data_user2idx)
test_data['Joke_unique'] = test_data['JokeId'].map(test_data_book2idx)

### Convert back to 3-column df
train_data = train_data[['User_unique', 'Joke_unique', 'Rating']]
test_data = test_data[['User_unique', 'Joke_unique', 'Rating']]
print(test_data.shape)

### TRAINING SET
# Create user-item matrices
n_users = train_data['User_unique'].nunique()
n_jokes = train_data['Joke_unique'].nunique()

print(train_data['User_unique'].unique())
print(train_data['User_unique'].count())
# First, create an empty matrix of size USERS x JOKES (this speeds up the later steps)
train_matrix = np.zeros((n_users, n_jokes))

# Then, add the appropriate vals to the matrix by extracting them from the df with itertuples
for entry in train_data.itertuples(): # entry[1] is the user-id, entry[2] is the book-isbn
    train_matrix[entry[1]-1, entry[2]-1] = entry[3] # -1 is to counter 0-based indexing

### TESTING SET
# Create user-item matrices
n_users = test_data['User_unique'].nunique()
n_jokes = test_data['Joke_unique'].nunique()

# First, create an empty matrix of size USERS x BOOKS (this speeds up the later steps)
test_matrix = np.zeros((n_users, n_jokes))

# Then, add the appropriate vals to the matrix by extracting them from the df with itertuples
for entry in test_data.itertuples(): # entry[1] is the user-id, entry[2] is the book-isbn
    test_matrix[entry[1]-1, entry[2]-1] = entry[3] # -1 is to counter 0-based indexing

#  It may take a while to calculate, so I'll perform on a subset initially
train_matrix_small = train_matrix[:25000, :25000]
test_matrix_small = test_matrix[:25000, :25000]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_matrix_small, metric='cosine')
item_similarity = pairwise_distances(train_matrix_small.T, metric='cosine') # .T transposes the matrix (NumPy)

def predict(ratings, similarity, type='user'): # default type is 'user'
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has the same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train_matrix_small, item_similarity, type='item')
user_prediction = predict(train_matrix_small, user_similarity, type='user')

from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(prediction, test_matrix):
    prediction = prediction[test_matrix.nonzero()].flatten()
    test_matrix = test_matrix[test_matrix.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, test_matrix))

# Call on test set to get error from each approach ('user' or 'item')
print(f'User-based CF RMSE: {rmse(user_prediction, test_matrix_small)}')
print(f'Item-based CF RMSE: {rmse(item_prediction, test_matrix_small)}')
