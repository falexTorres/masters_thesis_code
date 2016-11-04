from pylab import *
from numpy import *
from read_mnist import *

images, labels = read_mnist('training', digits=[0])

output = ""

csv_file = open("training_images_mnist.csv", "w")

for i in range(0,images.shape[0]):
	for j in range(0,images.shape[1]):
		for k in range(0,images.shape[2]):
			if k == (images.shape[0] - 1):
				output += str(images[i][j][k]) + "\n"
			else:
				output += str(images[i][j][k]) + ","

csv_file.write(output) 		
csv_file.close()

print images.shape
print 
print labels.shape