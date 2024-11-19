import numpy as np
from numpy import linalg as LA

def pca(matrix):
    matrix = np.array(matrix)
    centered = matrix - np.mean(matrix, axis = 0)
    covariance = np.cov(centered, rowvar = False)
    print(f"This is the covariance matrix: \n {covariance} \n")
    eigenvalues, eigenvectors = LA.eig(covariance)
    print(f"This is the set of eigenvalues: \n {np.round(eigenvalues,2)} \n")
    print(f"This is the set of eigenvectors (along the columns): \n {np.round(eigenvectors, 2)} \n")
    max1 = 0
    max2 = 0
    sum = 0
    for x in eigenvalues:
        sum = sum + x
        if x > max1:
            max1 = x
        elif x > max2:
            max2 = x
        else:
            continue
    max1_percent = np.round(max1/sum,3)*100
    max2_percent = np.round(max2/sum, 3)*100
    print(f"The two greatest eigenvalues for the covariance matrix are {np.round(max1,2)} ({max1_percent}%) and {np.round(max2,2)} ({max2_percent}%). Their corresponding eigenvectors are your principle components.")

print("Hello! This code can be used for principal component analysis. \n")
matrix_input = input("Please input a valid matrix, with entries along rows separated by commas and rows separated by spaces (e.x. 1,8 2,3): ")
rows_input = matrix_input.split()
matrix = [list(map(float, row.split(','))) for row in rows_input]
matrix_test = np.array(matrix)
print(f"This is your matrix: \n {matrix_test} \n")
pca(matrix_test)
