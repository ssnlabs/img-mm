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

# weights - https://www.kaggle.com/datasets/shivam316/yolov3-weights
# cfg - https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
# coco.names - https://www.kaggle.com/datasets/rajeevsinghsisodiya/yolo-model-and-configuration-files

import cv2
import numpy as np
import matplotlib.pyplot as plt

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

img = cv2.imread("image.jpg")
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), font, 1, color, 2)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis("off")
plt.show()


