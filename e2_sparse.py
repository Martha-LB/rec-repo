import numpy as np

from scipy.sparse import csr_matrix

def center(matrix):
    """ Center the matrix """
    if not isinstance(matrix, csr_matrix):
        raise TypeError("Input Error.")
    
    matrix = matrix.astype(float)

    if matrix.shape[0] == 1 or matrix.shape[1] == 1:
        mean_value = matrix.data.mean().astype(float)
        # print(f"mean is {mean_value}")
        if matrix.nnz == 0:
            return matrix
        centered_data = matrix.copy()
        nonzero_indices = centered_data.nonzero()[1] if matrix.shape[0] == 1 else centered_data.nonzero()[0]
        dense_values = centered_data[0, nonzero_indices].toarray().flatten().astype(float) if matrix.shape[0] == 1 else centered_data[nonzero_indices, 0].toarray().flatten().astype(float)
        
        # subtract
        dense_values -= mean_value

        # Assign the updated value back to the sparse matrix
        if matrix.shape[0] == 1:
            centered_data[0, nonzero_indices] = dense_values
        else:
            centered_data[nonzero_indices, 0] = dense_values
        return centered_data
    
    else:
        # users_means = np.array(matrix.mean(axis=0)).flatten()
        # print(f"users_means is {users_means}")
        centered_matrix = matrix.copy()
        for user in range(centered_matrix.shape[1]):
            user_mean = centered_matrix[:,user].data.mean().astype(float)
            nonzero_indices = centered_matrix[:, user].nonzero()[0] 

            if len(nonzero_indices) == 0:
                continue

            # Extract slices from sparse matrices, convert to dense arrays
            dense_values = centered_matrix[nonzero_indices, user].toarray().flatten()
        
            # Performs subtraction on dense arrays
            dense_values -= user_mean

            # tocsc
            # Update the values in the original matrix
            for idx, value in zip(nonzero_indices, dense_values):
                centered_matrix[idx, user] = value
        
        print("Centered matrix:\n", centered_matrix)

        return centered_matrix



def centered_cosine_sim(vector1, vector2):
    dot = vector1.dot(vector2.T).A[0, 0]
    norm1 = np.sqrt(vector1.power(2).sum())
    norm2 = np.sqrt(vector2.power(2).sum())
    result = dot / (norm1*norm2)
    print(f"Centered cosine similarity: {result}")
    return result

def fast_centered_cosine_sim(matrix, vector,axis = 0):
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)
    if not isinstance(vector, csr_matrix):
        vector = csr_matrix(vector)
    dot = vector.dot(matrix).A
    col_norms = np.sqrt(matrix.power(2).sum(axis=axis))
    col_norms = np.array(col_norms).flatten()
    norm = np.linalg.norm(vector.toarray())
    result = dot/(norm*col_norms)
    print(f"Centered cosine similarity: {result}")
    return result

##################test 1####################
np.random.seed(8)
matrix = np.random.randint(0, 3, size=(5,5))
vector1 = np.random.randint(0, 3, size=5)
vector2 = np.random.randint(0, 3, size=5)
vector1 = center(csr_matrix(vector1))
vector2 = center(csr_matrix(vector2))
vector2 = center(csr_matrix(matrix))
print("\n>>>result of test 1: ")
print(f"matrix1: {matrix}\n vector1: {vector1}\n vector2: {vector2}\n")
centered_cosine_sim(vector1, vector2)
fast_centered_cosine_sim(matrix, vector1)


###################test 2######################
x1 = np.array([])
y = np.array(range(100))

for i in range(100):
    x1 = np.append(x1,i+1)
    y[99-i] = i+1
x1 = center(csr_matrix(x1))
y = center(csr_matrix(y))
print("\n>>>result of test 2: ")
centered_cosine_sim(x1,y)


###################test 3######################
c = np.array([2, 3, 4, 5, 6])
increments = np.arange(0, 100, 10)
# Use broadcast to add increments to c
result = c[:, np.newaxis] + increments
# flatten!!!!
c = result.flatten()
x2 = np.array([])
for i in range(100):
    if i in c:
        x2 = np.append(x2,np.nan)
    else:
        x2 = np.append(x2,i+1)
x2 = center(csr_matrix(np.nan_to_num(x2)))
print("\n>>>result of test 3: ")
centered_cosine_sim(x2,y)