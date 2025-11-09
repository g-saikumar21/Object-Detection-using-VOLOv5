import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your custom YOLOv5 model (adjust path to weights file as needed)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', source='github')

# Read image and convert for YOLOv5 inference
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run inference
results = model(image_rgb)

# Extract results using pandas DataFrame
df = results.pandas().xyxy[0]

# Draw bounding boxes if confidence > 0.3
for idx, row in df.iterrows():
    if row['confidence'] > 0.3:
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{row['name']}, {row['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# Show the result using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
