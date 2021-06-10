#!/usr/bin/env python
# coding: utf-8

'''Book Recommender System
This project uses the [goodbooks-10K dataset](https://github.com/zygmuntz/goodbooks-10k) to build a similarity based recommendation system.
It uses both content based filtering based on book features and collaborative filtering to see how the results vary'''

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import pickle
from sklearn.preprocessing import MinMaxScaler

num_recos = 8
n_comp = 15
books = []
rating = []
def main():
     global rating
     global books
     # Load Data
     rating = pd.read_csv('../data/raw/goodreads/ratings.csv')
     books = pd.read_csv('../data/raw/goodreads/books.csv')
     tags = pd.read_csv('../data/raw/goodreads/tags.csv')
     book_tags = pd.read_csv('../data/raw/goodreads/book_tags.csv')

     # clean data and select faetures
     books = clean_books(books)
     (features, features_with_name) = get_features(book_tags, tags)

     # Train Content based filtering model
     nbrs = train_model(features)

     ratings_reduced = get_ratings(rating)
     # Train Collaborative filtering model
     cf_nbrs = train_model_cf(ratings_reduced)
     
     # predict similiar books for each book in our data set and store in df
     distances, indices = nbrs.kneighbors(features)
     reco_genre = pd.concat([books['original_title'],pd.DataFrame(indices, columns =['index1','index2','index3','index4','index5','index6','index7','index8'])], axis=1)

     distances, indices = cf_nbrs.kneighbors(ratings_reduced)
     reco_users = pd.concat([books['original_title'],pd.DataFrame(indices, columns =['index1','index2','index3','index4','index5','index6','index7','index8'])], axis=1)

     reco_users.to_pickle("output/reco_users.pkl")
     reco_genre.to_pickle("output/reco_genre.pkl")
     

def get_features(book_tags, tags):
     ''' Calculates features for each book including genres, avg rating and total number of ratings
     returns: X - a dataframe with the features for each book in the dataset '''

     book_features = books.drop(['book_id','work_id','best_book_id','books_count','isbn','isbn13','authors','original_publication_year','title','language_code', 'work_text_reviews_count','ratings_1','ratings_2'
     ,'ratings_3','ratings_4','ratings_5','image_url','small_image_url','ratings_count'], axis=1)

     book_tags = book_tags.merge(tags, on='tag_id').groupby('goodreads_book_id').tag_name.agg([('tag_name', ', '.join)])
     # add genres as features
     book_features = book_features.merge(book_tags, on='goodreads_book_id')
     book_features['fiction'] = book_features['tag_name'].str.contains('fiction')
     book_features['fantasy'] = book_features['tag_name'].str.contains('fantasy')
     book_features['young-adult'] = book_features['tag_name'].str.contains('young-adult|ya')
     book_features['romance'] = book_features['tag_name'].str.contains('romance')
     book_features['classics'] = book_features['tag_name'].str.contains('calssics')
     book_features['mystery'] = book_features['tag_name'].str.contains('mystery')
     book_features['sci-fi'] = book_features['tag_name'].str.contains('sci-fi|science-fiction')
     book_features['non-fiction'] = book_features['tag_name'].str.contains('non-fiction')
     book_features = book_features*1
     # scale rating scount
     # scale numeric values
     scaler = MinMaxScaler(feature_range=(0,5))
     book_features[['work_ratings_count']] = scaler.fit_transform(book_features[['work_ratings_count']])

     X = book_features.drop(['goodreads_book_id','original_title','tag_name'], axis=1)
     # X.to_pickle("data/features.pkl")

     return (X, book_features)

def train_model(features):
     ''' Train a nearest neighbor model to find the {num_recos} nearest neighbors to each data point in the features matrix'''
     nbrs = NearestNeighbors(n_neighbors=num_recos).fit(features)
     return nbrs

def clean_books(books):
     ''' This function removes all non English language books from our data
     returns: books - dataframe with only English books'''
     books = books.loc[books['language_code'].str.contains('eng|en-US|en-CA|en-GB|en').fillna(False)]
     books = books.reset_index(drop=True)
     books.to_pickle("output/books.pkl")
     return books


def get_ratings(rating):
     ''' Finds latent features based on book ratings using SVD
     returns: ratings_reduced - dataframe with a feature dimension of {n_comp}'''

     # remove non english books
     rating = rating.loc[rating['book_id'].isin(books['book_id'].values)]
     ratings_matrix = rating.pivot(index='book_id', columns='user_id', values='rating').fillna(0)

     svd = TruncatedSVD(n_components=n_comp)
     ratings_reduced = svd.fit_transform(ratings_matrix)
     return ratings_reduced

def train_model_cf(ratings):
     cf_nbrs = NearestNeighbors(n_neighbors=num_recos).fit(ratings)
     return cf_nbrs


if __name__ == '__main__':
    main()
