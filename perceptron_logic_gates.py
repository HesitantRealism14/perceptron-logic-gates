import seaborn as sns
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

data = [[0,0],[0,1],[1,0],[1,1]]
labels_AND = [0,0,0,1]
labels_OR = [0,1,1,1]


#AND gate
classifier = Perceptron()
classifier.fit(data, labels_AND)

x_values = np.linspace(0,1,100)
y_values = np.linspace(0,1,100)
point_grid = list(product(x_values,y_values))

distances = classifier.decision_function(point_grid)
abs_distances = [abs(distance) for distance in distances]
distances_matrix = np.reshape(abs_distances,(100,100))

heatmap = plt.pcolormesh(x_values,y_values,distances_matrix)

plt.colorbar(heatmap)
plt.title('AND gate')
plt.show()

#OR gate
classifier = Perceptron()
classifier.fit(data, labels_OR)

x_values = np.linspace(0,1,100)
y_values = np.linspace(0,1,100)
point_grid = list(product(x_values,y_values))

distances = classifier.decision_function(point_grid)
abs_distances = [abs(distance) for distance in distances]
distances_matrix = np.reshape(abs_distances,(100,100))

heatmap = plt.pcolormesh(x_values,y_values,distances_matrix)

plt.colorbar(heatmap)
plt.title('OR gate')
plt.show()