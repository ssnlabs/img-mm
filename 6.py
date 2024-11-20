#1
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"image.jpg", cv2.IMREAD_GRAYSCALE)

smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)

magnitude = cv2.magnitude(gradient_x, gradient_y)
angle = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(smoothed_image, low_threshold, high_threshold)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Canny Edge Detection')
plt.imshow(edges, cmap='gray')
plt.show()

#2
import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(img, seed, threshold):
    segmented_image = np.zeros_like(img)
    rows, cols = img.shape
    
    queue = [seed]
    seed_value = img[seed]
    segmented_image[seed] = 255
    
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while queue:
        x, y = queue.pop(0)
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and segmented_image[nx, ny] == 0:
                if abs(int(img[nx, ny]) - int(seed_value)) <= threshold:
                    queue.append((nx, ny))  
                    segmented_image[nx, ny] = 255  
    return segmented_image

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

seed_point = (50, 50)  
threshold_value = 30  

segmented_image = region_growing(image, seed_point, threshold_value)

plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image (Region Growing)')
plt.axis('off')

plt.show()



