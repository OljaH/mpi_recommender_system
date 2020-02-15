import pandas as pd
import pickle
import gensim
from gensim.test.utils import common_corpus, common_dictionary
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import *
from nltk.stem.porter import *
from sklearn import model_selection
import numpy as np
np.random.seed(400)

# 1. load the data
df_train_jokes = pd.read_csv("shortjokes.csv")
df_train_jokes = df_train_jokes.head(40000)
print(df_train_jokes.head())
print(df_train_jokes.shape)

# 2. data preprocessing functions
stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return result

# # example
#
# # '''
# # Preview a document after preprocessing
# # '''
# # document_num = 50
# # doc_sample = 'This disk has failed many times. I would like to get it replaced.'
# #
# # print("Original document: ")
# # words = []
# # for word in doc_sample.split(' '):
# #     words.append(word)
# # print(words)
# # print("\n\nTokenized and lemmatized document: ")
# # print(preprocess(doc_sample))

processed_docs = []

for doc in df_train_jokes['Joke']:
    processed_docs.append(preprocess(doc))

# 3. create bag of words
dictionary = gensim.corpora.Dictionary(processed_docs)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


document_num = 30
bow_doc_x = bow_corpus[document_num]
print(bow_corpus[10])
#
# for i in range(len(bow_doc_x)):
#     print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0],
#                                                      dictionary[bow_doc_x[i][0]],
#                                                      bow_doc_x[i][1]))
#
#
# LDA mono-core -- fallback code in case LdaMulticore throws an error on your machine
lda_model = gensim.models.LdaModel(bow_corpus,
                                   num_topics = 5,
                                   id2word = dictionary,
                                   passes = 100, decay=0.9)
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")


from gensim.test.utils import datapath

# Save model to disk.
temp_file = datapath("model_optimized")
lda_model.save(temp_file)

# Load a potentially pretrained model from disk.
df_test_jokes = pd.read_csv("JokeText.csv")
if False:
    lda_model = gensim.models.LdaModel.load(temp_file)
    keywords_score=[]
    if True:
        with open('topics_optimized.txt', 'w') as f:
            # f.write('Most important topics:\n')
            for index, row in df_test_jokes.iterrows():
                unseen_document=row['JokeText']
                #Data preprocessing step for the unseen document
                bow_vector = dictionary.doc2bow(preprocess(unseen_document))

                f.write(str(row['JokeId'])+"\n")
                # sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
                topic_scores=[0, 0, 0, 0, 0]
                for index, score in lda_model[bow_vector]:
                    f.write("Score: {}\t Topic {}: {}".format(score, index, lda_model.print_topic(index, 10))+"\n")
                    topic_scores[index]=score
                keywords_score.append(topic_scores)

                f.write("\n")

    print(len(keywords_score))
    print(len(keywords_score[0]))

    topics = lda_model.print_topics(num_words=10)
    for topic in topics:
        print(topic)

    # euklidova daljina
    def distance(lista, listb):
        return sum( (b - a) ** 2 for a,b in zip(lista, listb) ) ** .5

    matrix=[]

    for i in range(0, 100):
        row=[]
        for j in range(0, 100):
            row.append(1-distance(keywords_score[i], keywords_score[j]))
        matrix.append(row)

    with open('similarity_matrix_optimized', 'wb') as f:
        pickle.dump(matrix, f)

with open('similarity_matrix_optimized', 'rb') as f:
     mat = pickle.load(f)

def recommendations(JokeId, cosine_sim=mat):
    recommended_jokes = []

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[JokeId]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    print(score_series[1:10])
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_jokes.append(list(df_test_jokes.index)[i])

    return recommended_jokes

joke_number=89
print(df_test_jokes['JokeText'].iloc[[joke_number]])
print("************************")
for id in recommendations(joke_number)[0:2]:
    print(df_test_jokes['JokeText'].iloc[[str(id)]])