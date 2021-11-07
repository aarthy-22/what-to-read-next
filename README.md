Book Recommendation System
==============================

This is a book recommendation system using the [goodreads-100K dataset](https://github.com/zygmuntz/goodbooks-10k). On entering a book you liked, the algorithm recommends other books you might be interested in. 

Both content based and collaborative filtering are used to compare the results.

The project contains a jupyter notebook which includes EDA and other processing. It has also been converted into a Flask webapp.
See the live app here - https://stark-brook-74726.herokuapp.com/

![Recommended books](/images/recommend.png.png)

To run the app, run 'train.py' first to compute the recommendations and then run app.py to start the webapp

