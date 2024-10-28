# exercise 5

from e4_2 import load_user_vectors, rated_by
from scipy.sparse import coo_array,csr_matrix,coo_matrix,lil_matrix
from e3_sparse import rate_all_items
from memory_profiler import memory_usage



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
            ratings_matrix = lil_matrix((len(neighbors), 25000000))

            for i, user in enumerate(neighbors):
                if user in item_vectors:
                    user_vector = item_vectors[user]
                    ratings_matrix[i, :user_vector.shape[1]] = user_vector.toarray().flatten()

            
            predicted_rating_array = rate_all_items(ratings_matrix, int(user_id), neighborhood_size=1)
            predicted_rating = predicted_rating_array[item_id] if item_id in predicted_rating_array else None

    return predicted_rating

def measure_memory(user_id, item_id):
    """ Monitoring memory usage """
    mem_usage = memory_usage((predict_rating_for_user_item, (user_id, item_id)))
    return max(mem_usage)


if __name__ == '__main__':
    # Predict the rating of a given user
    # And then I realized I couldn't quite figure it out
    user_id_to_predict = '828'
    item_id_to_predict = '11'
    predicted_ratings = predict_rating_for_user_item(user_id_to_predict,item_id_to_predict)

    print(f"Predicted ratings for movie_id {item_id_to_predict} rated by user {user_id_to_predict} : {predicted_ratings}")

    # Getting Maximum Memory Usage
    max_memory_usage = measure_memory(user_id_to_predict, item_id_to_predict)
    print(f"Maximum Memory Usage for Rating Functions: {max_memory_usage:.2f} MiB")