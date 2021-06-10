from flask import Flask, render_template, request
from markupsafe import escape
import recommend

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    no_data = False
    (book_list, genre_book_list) = recommend.get_book_list(request.form['book_name'])
    if len(book_list)<1 or len(genre_book_list)<1:
        no_data = True
    return render_template('home.html', book_list=book_list, genre_book_list=genre_book_list, no_data=no_data)
    