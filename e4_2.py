# exercise 4

import numpy as np
import polars as pl
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import csv
from collections import defaultdict
import numpy as np
import shelve
from scipy.sparse import coo_array,csr_matrix,coo_matrix,lil_matrix
from e3_sparse import rate_all_items
from config import ConfigLf
import pickle


def load_movielens_tf(config):
    """ Load the MovieLens dataset using TensorFlow Datasets and preprocess it
        See https://www.tensorflow.org/ranking/tutorials/quickstart for alternative processing
    """

    postfix = '-ratings'
    ratings, info = tfds.load(config.dataset_base_name + postfix,
                              split=config.dataset_split,
                              shuffle_files=config.shuffle_files,
                              data_dir=config.data_dir,
                              with_info=True)
    print(
        f"Loaded dataset '{config.dataset_base_name}' with {ratings.cardinality()} ratings and features: {info.features}")
    assert isinstance(ratings, tf.data.Dataset)


# CSV file
RATINGS_FILE = '/private/tmp/movielens2/downloads/extracted/ZIP.files.grouple.org_dataset_moviele_ml-25miyHPt-sXBrTsCqyJQ2jZCs8m69-2rO0-vUrVvR65xqo.zip/ml-25m/ratings.csv'

rated_by = {}
user_col = {}

def read_ratings_file(ratings_file):
    """Reads the CSV file line by line and populates rated_by and user_col to store only valid ratings"""
    chunk_size = 100000  # Read less at once.
    for chunk in pd.read_csv(ratings_file, chunksize=chunk_size):
        for _, row in chunk.iterrows():
            user_id = str(int(row['userId']))
            item_id = str(int(row['movieId']))
            rating = row['rating']

            # add user to rated_by
            if item_id not in rated_by:
                rated_by[item_id] = []
            rated_by[item_id].append(user_id)

            # Save sparse vectors for each user
            if user_id not in user_col:
                user_col[user_id] = {}

            # Save non-empty scores
            user_col[user_id][item_id] = rating
    print(f"Finished reading ratings. Total users: {len(user_col)}")

def save_user_col_to_disk(user_col):
    # Check if the data to be stored contains content
    if not user_col:
        print("Warning: `user_col` is empty. No data to save.")
        return
    # Convert each user vector to a sparse matrix
    user_sparse_vectors = {}
    for user_id, ratings in user_col.items():
        items = list(map(int, ratings.keys()))
        ratings_values = list(map(float, ratings.values()))
        sparse_vector = csr_matrix((ratings_values, ([0]*len(ratings_values), items)), shape=(1, 25000000))
        user_sparse_vectors[user_id] = sparse_vector 

    with open('user_col.pkl', 'wb') as f:
        pickle.dump(user_sparse_vectors, f)
    # confirmation
    print(f"Data for {len(user_sparse_vectors)} users saved to 'user_col.pkl' successfully.")

def load_user_vectors(user_ids):
    try:
        with open('user_col.pkl', 'rb') as f:
            user_col = pickle.load(f)
            print(f"Loaded {len(user_col)} users at all!\n")
    except FileNotFoundError:
        print("Hey can not find file 'user_col.pkl'.")
        return {}
    except Exception as e:
        print(f"Hey an error happened: {e}")
        return {}

    # Check if the required user data exists and load the vector for the specified user
    loaded_vectors = {}
    for user_id in user_ids:
        if user_id in user_col:
            loaded_vectors[user_id] = user_col[user_id]
        else:
            print(f"Hey user {user_id} not found in loaded data.")
    
    # Check the final number of users loaded
    if not loaded_vectors:
        print("Here was zero user vector can be found.")
    else:
        print(f"Loaded {len(loaded_vectors)} out of {len(user_ids)} requested users.")
    
    return loaded_vectors


def predict_rating_for_user_item(user_id, item_id):
    user_id = str(user_id)
    item_id = str(item_id)

    
    # Load the current user's vector
    current_user_vector = load_user_vectors([user_id]).get(user_id)
    
    # Check if the user has already rated the project
    if item_id in current_user_vector.indices:
        existing_rating = current_user_vector[0, current_user_vector.indices == int(item_id)]
        print(f"User {user_id} has already rated item {item_id} with a rating of {existing_rating[0]}.")
        return existing_rating[0]

    # Storage Prediction Score
    predicted_rating = None

    # Check if the user has already rated the project
    if item_id not in current_user_vector.indices:
        if item_id in rated_by:
            neighbors = rated_by[item_id]
            item_vectors = load_user_vectors(neighbors)

            # Creating temporary sparse matrices
            ratings_matrix = lil_matrix((len(neighbors), 25000000))  # 使用 lil_matrix 便于后续赋值

            for i, user in enumerate(neighbors):
                if user in item_vectors:
                    user_vector = item_vectors[user]
                    ratings_matrix[i, :user_vector.shape[1]] = user_vector.toarray().flatten()

            predicted_rating_array = rate_all_items(ratings_matrix, int(user_id), neighborhood_size=1)
            predicted_rating = predicted_rating_array[item_id] if item_id in predicted_rating_array else None

    return predicted_rating


if __name__ == '__main__':
    load_movielens_tf(ConfigLf)

    # Reading rating files and constructing data structures
    read_ratings_file(RATINGS_FILE)

    # save
    save_user_col_to_disk(user_col)

    # predict
    user_id_to_predict = '8'
    item_id_to_predict = '18'
    predicted_ratings = predict_rating_for_user_item(user_id_to_predict,item_id_to_predict)

    print(f"Predicted ratings for item{item_id_to_predict} rated by user{user_id_to_predict}: {predicted_ratings}")


