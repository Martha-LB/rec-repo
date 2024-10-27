# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import numpy as np
from scipy.sparse import csr_matrix
from e2_sparse import center

def centered_cosine_sim(vector1, vector2):
    dot = vector1.dot(vector2.T).A[0, 0]
    norm1 = np.sqrt(vector1.power(2).sum())
    norm2 = np.sqrt(vector2.power(2).sum())
    result = dot / (norm1*norm2)
    print(f"Centered cosine similarity: {result}")
    return result

def fast_centered_cosine_sim(matrix, vector,axis = 0):
    vector_transposed = vector.T
    dot = vector_transposed.dot(matrix).A
    col_norms = np.sqrt(matrix.power(2).sum(axis=axis))
    col_norms = np.array(col_norms).flatten()
    norm = np.linalg.norm(vector.toarray())
    result = dot/(norm*col_norms)
    return result.flatten()

#########################################



# Implement the CF from the lecture 1
def rate_all_items(orig_utility_matrix, user_index ,neighborhood_size):
    """Predicts ratings for all items that have not yet been rated for a given user"""
    print(">>>CF computation for UM w\n shape: "
          + f"{orig_utility_matrix.shape}\n user_index: {user_index}\n neighborhood_size: {neighborhood_size}\n")
    """ Compute the rating of all items not yet rated by the user"""
    # centralized matrix
    clean_utility_matrix = center(orig_utility_matrix)
    # Compute the cosine similarity between the user and all other users
    user_col = clean_utility_matrix[:, user_index]
    similarities = fast_centered_cosine_sim(clean_utility_matrix, user_col)
    print(f"similarities: {similarities}\n")

    def rate_one_item(item_index):
        """ predict rating of item_index """
        # If the user has already rated the item, return the rating
        if not np.isnan(orig_utility_matrix[item_index, user_index]):
            if orig_utility_matrix[item_index, user_index] != 0:
                return orig_utility_matrix[item_index, user_index]

        # Find the indices of users who rated the item
        users_who_rated = orig_utility_matrix[item_index, :].nonzero()[1]  # Get non-zero indices
        print(f"users_who_rated: {users_who_rated}\n")

        # From those, get indices of users with the highest similarity
        best_among_who_rated = np.argsort(similarities[users_who_rated])
        print(f"similarities[users_who_rated]: {similarities[users_who_rated]}\n")
        print(f"best_among_who_rated: {best_among_who_rated}\n")

        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighborhood_size:]
        print(f"best_among_who_rated after cut: {best_among_who_rated}\n")

        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]
        # Retain only those indices where the similarity is not nan
        best_among_who_rated = best_among_who_rated[~np.isnan(similarities[best_among_who_rated])]
    
        if best_among_who_rated.size > 0:
            # Compute the rating of the item
            neighbors_ratings = orig_utility_matrix[item_index, best_among_who_rated].toarray().flatten()  # Convert to dense array if needed
            neighbors_similarities = similarities[best_among_who_rated]
            rating_of_item = np.dot(neighbors_ratings, neighbors_similarities) / np.sum(np.abs(neighbors_similarities))
        else:
            rating_of_item = np.nan
    
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item
    num_items = orig_utility_matrix.shape[1]
    print(range(num_items))
    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items))) # Applying a function to each element of a list
    return ratings

if __name__ == '__main__':
    np.random.seed(41)
    orig_utility_matrix = np.random.randint(1, 6, size=(5, 5)).astype(float)
    mask = np.random.choice([1, 0], orig_utility_matrix.shape, p=[0.7, 0.3])
    orig_utility_matrix[mask == 0] = np.nan
    print(">>>Original Utility Matrix:\n", orig_utility_matrix)

    user_index = 1
    orig_utility_matrix = csr_matrix(np.nan_to_num(orig_utility_matrix))
    print(f"clean UM: {orig_utility_matrix}")
    # set neighborhood_size
    neighborhood_size = 1

    ratings = rate_all_items(orig_utility_matrix, user_index ,neighborhood_size)
    print("\nPredicted Ratings for User:", ratings)

