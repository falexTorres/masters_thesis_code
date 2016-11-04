import numpy as np
import matplotlib.pyplot as plt

csv_file = open('training_images_mnist.csv')

images = np.zeros((5923, 28, 28), dtype='uint8')

for i in range(0, 5923):
	line = csv_file.readline()
	for j in range(0, 28):
		for k in range(0, 28):
			elements = line.split(',')
			images[i][j][k] = elements[k]

print show[0][0][0]

plt.xlabel(label)
imshow(images[0], cmap=cm.gray)
show()