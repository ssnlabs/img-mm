import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'image.jpg'
image = cv2.imread(image_path,cv2.IMREAD_COLOR)

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# 1. Scaling
scaled_image = cv2.resize(image, None, fx=1.5, fy=1.5)

plt.subplot(2, 2, 2)
plt.imshow(scaled_image)
plt.title("Scaled Image (50%)")
plt.axis('off')

# 2. Rotation
angle = 45 
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

plt.subplot(2, 2, 3)
plt.imshow(rotated_image)
plt.title("Rotated Image (45 degrees)")
plt.axis('off')

# 3. Shearing
shear_factor = 0.2
M_shear = np.array([[1, shear_factor, 0],
                    [shear_factor, 1, 0],
                    [0, 0, 1]], dtype=float)

sheared_image = cv2.warpPerspective(image, M_shear, (int(w * (1 + shear_factor)), int(h * (1 + shear_factor))))

plt.subplot(2, 2, 4)
plt.imshow(sheared_image)
plt.title("Sheared Image")
plt.axis('off')

# Display all images
plt.tight_layout()
plt.show()
