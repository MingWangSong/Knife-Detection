import numpy as np
from skimage import data
import matplotlib.pyplot as pylab
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import morphology
import cv2


image_address = r'../images/source/noclean-040.png'
coins = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)
hist = np.histogram(coins, bins=np.arange(0, 256), normed=True)
edges = canny(coins, sigma=2)
fill_coins = ndi.binary_fill_holes(edges)
# coins_cleaned = morphology.remove_small_objects(fill_coins, 1)
coins_cleaned = fill_coins
fig, axes = pylab.subplots(figsize=(10, 6))
axes.imshow(coins_cleaned, cmap=pylab.cm.gray, interpolation='nearest')
axes.axis('off'), pylab.show()