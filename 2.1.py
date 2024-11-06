import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'image.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
min_val = np.min(image)
max_val = np.max(image)

stretched_image = ((image - min_val) / (max_val - min_val)) * 255
filtered_image = cv2.GaussianBlur(image, (5, 5), 0)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(stretched_image, cmap='gray')
plt.title('Contrast Stretched Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Gaussian Blur)')
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
plt.subplot(2,2, 1)
plt.hist(image.ravel(), bins=256, range=(0, 255), color='gray')
plt.title('Histogram of Original Image')

plt.subplot(2,2, 2)
plt.hist(stretched_image.ravel(), bins=256, range=(0, 255), color='gray')
plt.title('Histogram of Contrast Stretched Image')

plt.subplot(2,2, 3)
plt.hist(filtered_image.ravel(), bins=256, range=(0, 255), color='gray')
plt.title('Histogram of Filtered Image')

plt.tight_layout()
plt.show()