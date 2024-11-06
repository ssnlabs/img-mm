import cv2
import numpy as np
image1_path = r'horse.jpg'
image2_path = r'image.jpg'
M1 = cv2.imread(image1_path)
M2 = cv2.imread(image2_path)
print(M1.shape)
M2 = cv2.resize(M2, (M1.shape[1], M1.shape[0]))
Out = cv2.absdiff(M1, M2)
cv2.imshow('Absolute Difference', Out)
cv2.waitKey(0)
cv2.destroyAllWindows()
