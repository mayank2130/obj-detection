import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torchvision.transforms as T
import time
from datetime import datetime, timedelta

def manual_preprocess(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    transform = T.Compose([
        T.Resize((800, 1333)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def process_frame(frame, model, processor):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    preprocessed_image = manual_preprocess(image)
    
    with torch.no_grad():
        outputs = model(preprocessed_image)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
    
    return results

def count_vehicles(results, vehicle_classes):
    return sum(1 for label in results["labels"] if model.config.id2label[label.item()] in vehicle_classes)

# Load model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
print("Model and processor loaded successfully")

# Open video file
video_path = "your_video_path.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Initialize variables
vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}
total_count = 0
interval_count = 0
start_time = time.time()
interval_start = start_time
interval_duration = 30 * 60  # 30 minutes in seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_time = time.time()
    
    # Process frame and count vehicles
    results = process_frame(frame, model, processor)
    frame_count = count_vehicles(results, vehicle_classes)
    total_count += frame_count
    interval_count += frame_count
    
    # Check if 30 minutes have passed
    if current_time - interval_start >= interval_duration:
        interval_end = datetime.fromtimestamp(interval_start + interval_duration)
        interval_start_time = datetime.fromtimestamp(interval_start)
        
        print(f"{interval_start_time.strftime('%H:%M')} - {interval_end.strftime('%H:%M')} ----- Count of vehicles is {interval_count}")
        
        # Reset for next interval
        interval_count = 0
        interval_start = current_time

    # Optional: Display frame with detections
    # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    #     if score > 0.7 and model.config.id2label[label.item()] in vehicle_classes:
    #         box = [int(i) for i in box.tolist()]
    #         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #         cv2.putText(frame, f"{model.config.id2label[label.item()]}: {score:.2f}", (box[0], box[1] - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # 
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()

print(f"Total vehicles detected: {total_count}")