import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import csv
import time

def pca(matrix,labels,unique_types,name,standardized_yes_no):
    matrix = np.array(matrix)
    centered = matrix - np.mean(matrix, axis = 0, keepdims=True)
    print(f"This is the centered matrix: \n {centered} \n")
    if standardized_yes_no == "y":
        std_dev = np.std(matrix,axis=0,ddof=1)
        standardized = centered/std_dev
        print(f"This is the standardized matrix: \n {standardized} \n")
        covariance = np.cov(standardized, rowvar = False)
    else:
        covariance = np.cov(centered, rowvar=False)
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

    sorted_eigenvalues = np.argsort(eigenvalues)[::-1]
    pc1_index = sorted_eigenvalues[0]
    pc2_index = sorted_eigenvalues[1]
    xlist = []
    for i in range(len(centered)):
        value = 0
        for j in range(centered.shape[1]):
           value += centered[i,j]*eigenvectors[j,pc1_index]
        xlist.append(value)

    ylist = []
    for i in range(len(centered)):
        value = 0
        for j in range(centered.shape[1]):
           value += centered[i,j]*eigenvectors[j,pc2_index]
        ylist.append(value)

    unique_types = unique_types.tolist()
    colors = ['green', 'blue', 'red', 'purple', 'brown', 'orange', 'pink', 'cyan', 'magenta']
    colors.append(np.random.choice(['yellow', 'teal', 'navy', 'lime']))

    added_labels = set()
    for i in range(len(labels)):
        index = unique_types.index(labels[i])
        if labels[i] not in added_labels:
            plt.scatter(xlist[i], ylist[i], color=colors[index], label = unique_types[index])
            added_labels.add(labels[i])
        else:
            plt.scatter(xlist[i], ylist[i], color=colors[index])
    plt.xlabel(f"PC1 ({max1_percent}%)")
    plt.ylabel(f"PC2 ({max2_percent}%)")
    plt.title(name)
    plt.axvline(x=0, color='black')
    plt.axhline(y=0, color='red')
    plt.legend()
    plt.show()

def load_data(filename):
    labels = []
    list = []
    with open(filename) as file:
        genes = csv.reader(file, delimiter = ',')
        next(genes)
        for row in genes:
            labels.append(row[0])
            list.append([float(x) for x in row[1:]])
    return labels,list

print("Hello! This code for UM51A will perform PCA on a given dataset. \n")
title = input("Please input the title of your plot: ")
yes_no = input("Would you like your data to be standardized (y/n): ")
time.sleep(1)
labels, new_list = load_data('genes.csv')
matrix_test = np.array(new_list)
cell_types = matrix_test[:,0]
unique_cell_types = np.unique(labels)
print(f"This is your matrix: \n{matrix_test}\n")
pca(matrix_test,labels,unique_cell_types,title,yes_no)
