import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)

gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)

gradient_direction = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)

nms_image = cv2.Canny(smoothed_image, 100, 200)

plt.figure(figsize=(10, 10))
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(2, 3, 2)
plt.title("Smoothed Image")
plt.imshow(smoothed_image, cmap='gray')

plt.subplot(2, 3, 3)
plt.title("Gradient Magnitude")
plt.imshow(gradient_magnitude, cmap='gray')

plt.subplot(2, 3, 4)
plt.title("Gradient Direction")
plt.imshow(gradient_direction, cmap='gray')

plt.subplot(2, 3, 5)
plt.title("Canny Edge Detection (After NMS)")
plt.imshow(nms_image, cmap='gray')

plt.show()
