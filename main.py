import numpy as np
import time

def compare_dot_numpy(A, B):
    """
    For each row in A and column in B, count how many matching pairs exist
    when comparing elements like in matrix multiplication
    """
    m, _ = A.shape # where m is the number of rows in A
    _, p = B.shape # where p is the number of columns in B
    print(f'm: {m}, p: {p}')
    result = np.zeros((m, p)) # Initialize result matrix
    # Here, we iterate over each row in A and each column in B
    for i in range(m): # for each row in A
        for j in range(p): # for each column in B
            # Compare elements that would be multiplied in dot product
            comparison_result = (A[i] == B[j]).sum() # row i does element-wise comparison with row j
            result[i][j] = comparison_result
    return result.T


# In an effort to improve my genetic algorithm for the closest string problem,
# I plan to implement a more efficient fitness function. This will rely on matrix operations.
# The population of strings is represented as a 2D numpy array, where each row is a string.
# The target strings are represented as a 2D numpy array, where each row is a target string.

# Generate sample 1D array data that is very large.
size = 10**7 # 10 million
a = np.random.randint(0, 2, size)
b = np.random.randint(0, 2, size)
print(a)
print(f"Arrays A and B have {a.shape[0]} elements")

# NumPy vectorized comparison and sum
start = time.time()
result_numpy = (a == b).sum()
numpy_time = time.time() - start
print("Comparing arrays with Numpy, 10-million elements:")
print(f"NumPy vectorized time: {numpy_time:.4f} seconds")
print(f"Matching elements: {result_numpy}")

# A very small example that's easy to follow, 3x3 matrix.
x = np.array([[1, 0, 1],
             [1, 1, 0],
             [0, 0, 0]]
             )

y = np.array([[1, 0, 1], # Total fitness score: 5
             [0, 1, 0], # Total fitness score: 4
             [1, 1, 1]] # Total fitness score: 4
             )

# NumPy vectorized comparison and sum
#start = time.time()
#result_numpy = compare_dot_numpy(x, y)
#numpy_time = time.time() - start
#print("\n\nComparing 3x3 arrays with Numpy:")
#print(f"NumPy vectorized time: {numpy_time:.4f} seconds")
#print(f"Matching elements (fitness score):\n {result_numpy}")


# Example matrix
size = 10**3  # 10^3 by 10^3 matrix
matrix_a = np.random.randint(0, 2, (size, size)) # Let's assume this is our target
matrix_b = np.random.randint(0, 2, (size, size)) # Let's assume this is our population

# NumPy vectorized comparison and sum
# The function that I wrote essentially computes Hamming distance between a population row and every target row.
# But this is not efficient, as it is linear to the number of rows and columns between the two matrices, thus O(m*p).
start = time.time()
result_numpy = compare_dot_numpy(matrix_a, matrix_b)
numpy_time = time.time() - start
print("\nComparing 2D arrays (matrices) with Numpy:")
print(f"NumPy vectorized time: {numpy_time:.4f} seconds")
print(f"Matching elements: {result_numpy}")

# A dot product is more efficient, but essentially captures a Jaccard similarity, since it counts the number of positive matches
start = time.time()
result_numpy = np.dot(matrix_a, matrix_b)
numpy_time = time.time() - start
print("\nComparing 2D arrays (matrices) with Numpy with matrix multiplication:")
print(f"NumPy vectorized time: {numpy_time:.4f} seconds")
print(f"Matching elements: {result_numpy}")


# 1's array of size 1000
# Taking the dot product of a 1's array and the resulting matrix effectively sums the rows.
# Summing the rows gives the total fitness score for each string in the population.
size = 10**3
ones_vector = np.ones(size)

start = time.time()
result_numpy = result_numpy.dot(ones_vector)
total_fitness = result_numpy.sum() # Think of this as a population's total fitness. In this case, we are selecting for the entire population's positive matches
numpy_time = time.time() - start
print("\nSumming the:")
print(f"NumPy vectorized time: {numpy_time:.4f} seconds")
print(f"Matching elements: {result_numpy}")
print(f"Total fitness score: {total_fitness}")


# Mutation may be performed with element-wise multiplication
# Let z be our mutation array
z = np.array([2, 1, 2])

# Mutate by element-wise multiplication followed by modulo 2
mutated_y = (y + z) % 2
print("\nMutated array:")
print(mutated_y)




# Define the size of the matrix; only need to do this once
rows, cols = 10**3, 10**3  # Example size
# Calculate the total number of elements to be set to 1; only need to do this once
total_elements = rows * cols
num_ones = int(0.10 * total_elements)


# Create mutation matrix with 2's
mutation_matrix = np.full((rows, cols), 2)
# Randomly select indices to set to 1
indices = np.random.choice(total_elements, num_ones, replace=False)
# Convert the flat indices to 2D indices
row_indices, col_indices = np.unravel_index(indices, (rows, cols))
# Set the selected indices to 1
mutation_matrix[row_indices, col_indices] = 1
print("\nMutation matrix:")
print(mutation_matrix)

#
# TESTING A SIMPLE EVOLUTIONARY ALGORITHM
#
#



start = time.time()
# Here, let's test a simple selection cycle.
matrix_a = np.random.randint(0, 2, (size, size)) # Let's assume this is our target
matrix_b = np.random.randint(0, 2, (size, size)) # Let's assume this is our population
_ = time.time()
original_fitness = np.dot(matrix_a, matrix_b).dot(ones_vector).sum()
endtime = time.time() - _
print(f"Original fitness: {original_fitness} \t Time: {endtime:.4f} seconds")
for _ in range(10):
    # Calculate fitness
    total_fitness = np.dot(matrix_a, matrix_b).dot(ones_vector).sum()

    mutation_matrix = np.full((rows, cols), 2)
    indices = np.random.choice(total_elements, num_ones, replace=False)
    row_indices, col_indices = np.unravel_index(indices, (rows, cols))
    mutation_matrix[row_indices, col_indices] = 1

    mutated_b = (matrix_b + mutation_matrix) % 2
    mutated_fitness = np.dot(matrix_a, mutated_b).dot(ones_vector).sum()

    #if (mutated_fitness > total_fitness).sum()/size > 0.50:
    if mutated_fitness > total_fitness:
        matrix_b = mutated_b
        #print(f"{(mutated_fitness > total_fitness).sum()/size:.2f} of the population improved. PASS!")
        print(f"original: {original_fitness} \t total fitness: {total_fitness} \t fitness after mutation: {mutated_fitness} \t Mutation successful! Re-selecting population...")
        continue

    print(f"original: {original_fitness} \t total fitness: {total_fitness} \t fitness after mutation: {mutated_fitness} \t No improvement.")
    #print(f"{(mutated_fitness > total_fitness).sum()/size:.2f} of the population improved. FAIL")

numpy_time = time.time() - start
print("\nSimple evolutionary algorithm:")
print(f"Time: {numpy_time:.4f} seconds")