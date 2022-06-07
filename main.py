import pandas as pd
import numpy as np
import string
import nltk
from flask import Flask, render_template, request
post_1_df=pd.read_csv('C://Users/USER/Downloads/post_1_df_id.csv')
# post_1_df.head()

post_1_df.drop(columns=['Product ID','Brand Name'],inplace=True)
post_1_df.fillna(" ",inplace=True)
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
stop_words = stopwords.words("english")
punc = string.punctuation
spec_chars = ["\\n","\\xa0","!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–","0","1","2","3","4","5","6","7","8","9"]

MIN_WORDS = 4
MAX_WORDS = 200
from nltk.stem import WordNetLemmatizer
def tokenizer(sentence, min_words=MIN_WORDS, max_words=MAX_WORDS,lemmatize=True):
    """
    Lemmatize, tokenize, crop and remove stop words.
    """
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
    else:
        tokens = [w for w in word_tokenize(sentence)]
    token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                                        and w not in stop_words)]
    return tokens
# post_1_df.head()

def extract_best_indices(m, topk, mask=None):
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0) 
    else: 
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]  
    return best_index

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Adapt stop words
token_stop = tokenizer(' '.join(stop_words), lemmatize=False)

# Fit TFIDF
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer) 
tfidf_mat = vectorizer.fit_transform(post_1_df['Data'].values) # -> (num_sentences, num_vocabulary)
# tfidf_mat.shape

def get_recommendations_tfidf(sentence):
    
    """
    Return the database sentences in order of highest cosine similarity relatively to each 
    token of the target sentence. 
    """
    # Embed the query sentence
    tokens = [str(tok) for tok in tokenizer(sentence)]
    vec = vectorizer.transform(tokens)
    # Create list with similarity between query and dataset
    mat = cosine_similarity(vec, tfidf_mat)
    # Best cosine distance for each token independantly
    print(mat.shape)
    best_index = extract_best_indices(mat, topk=10)
    return best_index

def search_tf(query):
    best_index = get_recommendations_tfidf(query)
    #l=[]
    df=pd.DataFrame(post_1_df[['Post Title']].iloc[best_index])
    df['post_id']=best_index
    l=df.values.tolist()
    return l

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = get_recommendations_tfidf(movie)
#     movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r)
    else:
        return render_template('recommend.html',movie=movie,r=r)



if __name__ == '__main__':
    app.run()