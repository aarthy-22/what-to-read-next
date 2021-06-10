import pickle
import numpy as np
import pandas as pd

books = pd.read_pickle("output/books.pkl")
reco_users = pd.read_pickle("output/reco_users.pkl")
reco_genre = pd.read_pickle("output/reco_genre.pkl") 

def get_book_list(book_name):
    book_list = []
    genre_book_list = []
    book = reco_genre.loc[reco_genre['original_title'].str.lower() == str(book_name).lower()]
    if not book.empty:
        reco_df = books.iloc[book.iloc[:,2:9].values[0]]
        genre_book_list = [(Books(row.original_title,row.authors, row.average_rating, row.goodreads_book_id, row.image_url)) for index, row in reco_df.iterrows() ] 

    book = reco_users.loc[reco_users['original_title'].str.lower() == str(book_name).lower()]
    if not book.empty: 
        reco_df = books.iloc[book.iloc[:,2:9].values[0]]
        book_list = [(Books(row.original_title,row.authors, row.average_rating, row.goodreads_book_id, row.image_url)) for index, row in reco_df.iterrows() ]  
    return (book_list, genre_book_list)


class Books: 
    def __init__(self, name, author, rating, url, img): 
        self.name = name 
        self.author = author
        self.rating = rating
        self.url = url
        self.img = img

