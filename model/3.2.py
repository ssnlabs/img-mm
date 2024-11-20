import cv2
import matplotlib.pyplot as plt

image_path = 'image.jpg'  
image = cv2.imread(image_path,cv2.IMREAD_COLOR)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image,None   )
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title("Image with SIFT Keypoints")
plt.axis('off')
plt.show()

print(f"Number of keypoints detected: {len(keypoints)}")
print("Descriptor shape:", descriptors.shape)
