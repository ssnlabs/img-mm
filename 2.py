#1
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(image, None)

image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10, 6))
plt.imshow(image_with_keypoints, cmap='gray')
plt.title('SIFT Keypoints')
plt.axis('off')
plt.show()

print(f"Number of keypoints detected: {len(keypoints)}")

print("Keypoint Coordinates (x, y):")
for i, kp in enumerate(keypoints[:5]):
    print(f"Keypoint {i+1}: ({kp.pt[0]}, {kp.pt[1]})")

print("\nOrientation of keypoints:")
for i, kp in enumerate(keypoints[:5]):
    print(f"Keypoint {i+1}: Orientation = {kp.angle} degrees")

print("\nDescriptor of first keypoint:")
print(descriptors[0])

print(f"\nLength of each descriptor vector: {descriptors.shape[1]}")

#2
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt') 

image_path = 'image.jpg' 
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model(image_rgb)

annotated_image = results[0].plot() 

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Object Detection using YOLOv8")
plt.show()