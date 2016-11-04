from pylab import *
from numpy import *
from read_mnist import *
import time

start_time = time.time()

images, labels = read_mnist('training', digits=[0,1,2,3,4,5,6,7,8,9])
#label = "label: " + str(labels[0])
#plt.xlabel(label)
#imshow(images[0], cmap=cm.gray)
#show()

print("--- %s seconds ---" % (time.time() - start_time))