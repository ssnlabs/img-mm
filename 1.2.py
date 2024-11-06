import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

image_path = r'image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (64, 128))
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0,ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1,ksize=3)
print("The gradients are : ", gradient_x, gradient_y)
magnitude = cv2.magnitude(gradient_x, gradient_y)  
orientation = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)  
print("Magnitude: ", magnitude)
print("Gradient: ",orientation)

features, hog_image = hog(image,visualize=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))

plt.figure()
plt.imshow(image, cmap="gray")
plt.title('Original Image')
plt.show()
plt.figure()
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title('HOG Image')
plt.show()

print(features)
print("HOG feature vector length:", len(features))
