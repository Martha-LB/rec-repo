# exercise 1
# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import numpy as np

def complete_code(message):
    raise Exception(f"Please complete the code: {message}")
    return None


def center_and_nan_to_zero(matrix, axis=0):
    """ Center the matrix and replace nan values with zeros"""
    # Compute along axis 'axis' the mean of non-nan values
    # E.g. axis=0: mean of each column, since op is along rows (axis=0)
    means = np.nanmean(matrix, axis=axis)
    # Subtract the mean from each axis
    matrix_centered = matrix - means
    return np.nan_to_num(matrix_centered)


def cosine_sim(u, v):

    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def fast_cosine_sim(utility_matrix, vector, axis=0):
    """ Compute the cosine similarity between the matrix and the vector"""
    # Compute the norms of each column
    norms = np.linalg.norm(utility_matrix, axis=axis)
    um_normalized = utility_matrix / norms
    # Compute the dot product of transposed normalized matrix and the vector
    # dot = complete_code("fast_cosine_sim")
    #####################################################
    dot = np.dot(np.transpose(vector),um_normalized)

    # Scale by the vector norm
    scaled = dot / np.linalg.norm(vector)
    return scaled


# Implement the CF from the lecture 1
def rate_all_items(orig_utility_matrix, user_index, neighborhood_size):
    """Predicts the ratings of all items that have not yet been rated for a given user"""
    print(f"\n>>> CF computation for UM w\n shape: "
          + f"{orig_utility_matrix.shape}\n user_index: {user_index}\n neighborhood_size: {neighborhood_size}\n")
    
    clean_utility_matrix = center_and_nan_to_zero(orig_utility_matrix)
    """ Compute the rating of all items not yet rated by the user"""
    user_col = clean_utility_matrix[:, user_index]
    # Compute the cosine similarity between the user and all other users
    similarities = fast_cosine_sim(clean_utility_matrix, user_col)
    print(f"similarities: {similarities}\n")

    def rate_one_item(item_index):
        """predict rating of item_index"""
        # If the user has already rated the item, return the rating
        if not np.isnan(orig_utility_matrix[item_index, user_index]):
            return orig_utility_matrix[item_index, user_index]

        # Find the indices of users who rated the item
        users_who_rated = np.where(np.isnan(orig_utility_matrix[item_index, :]) == False)[0]
        print(f"users_who_rated: {users_who_rated}\n")

        # From those, get indices of users with the highest similarity (watch out: result indices are rel. to users_who_rated)
        best_among_who_rated = np.argsort(similarities[users_who_rated])
        print(f"similarities[users_who_rated]: {similarities[users_who_rated]}\n")
        print(f"best_among_who_rated: {best_among_who_rated}\n")


        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighborhood_size:]
        print(f"best_among_who_rated after cut: {best_among_who_rated}\n")

        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]
        # Retain only those indices where the similarity is not nan
        best_among_who_rated = best_among_who_rated[np.isnan(similarities[best_among_who_rated]) == False]
        if best_among_who_rated.size > 0:
            # Compute the rating of the item
            neighbors_ratings = orig_utility_matrix[item_index, best_among_who_rated]
            neighbors_similarities = similarities[best_among_who_rated]
            rating_of_item = np.dot(neighbors_ratings, neighbors_similarities) / np.sum(abs(neighbors_similarities))
            # rating_of_item = complete_code("compute the ratings")
        else:
            rating_of_item = np.nan
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item

    num_items = orig_utility_matrix.shape[0] 

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items))) 
    return ratings


np.random.seed(41)
orig_utility_matrix = np.random.randint(1, 6, size=(5, 5)).astype(float)
mask = np.random.choice([1, 0], orig_utility_matrix.shape, p=[0.7, 0.3])
orig_utility_matrix[mask == 0] = np.nan


user_index = 1

neighborhood_size = 1

print(">>>Original Utility Matrix:\n", orig_utility_matrix)

ratings = rate_all_items(orig_utility_matrix, user_index, neighborhood_size)
print("\nPredicted Ratings for User:", ratings)

