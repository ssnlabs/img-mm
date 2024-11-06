import cv2
import os

video_path = 'path/to/your/video.mp4'  
output_folder = 'extracted_images'  

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break 
    frame_filename = os.path.join(output_folder, f"{frame_count}.jpg")
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

cap.release()
print("Image extraction completed!")
