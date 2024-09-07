from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import os
import torchvision.transforms as T

def print_image_info(image):
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")

def manual_preprocess(image):
    if image.mode == 'RGBA':
        print("Converting RGBA image to RGB")
        image = image.convert('RGB')
    
    transform = T.Compose([
        T.Resize((800, 1333)),  # DETR's default size
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

image_path = "./traffic.png"

if not os.path.exists(image_path):
    print(f"Error: File does not exist at {image_path}")
    exit()

try:
    image = Image.open(image_path)
    print("Image opened successfully")
    print_image_info(image)
except Exception as e:
    print(f"Error opening image: {e}")
    exit()

try:
    preprocessed_image = manual_preprocess(image)
    print(f"Manual preprocessing successful. Shape: {preprocessed_image.shape}")
except Exception as e:
    print(f"Error during manual preprocessing: {e}")
    exit()

try:
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    print("Model and processor loaded successfully")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit()

try:
    # Use the manually preprocessed image directly
    outputs = model(preprocessed_image)
    print("Model inference completed")

    # Post-process the outputs
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Print the detected objects
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
except Exception as e:
    print(f"Error during processing or inference: {e}")
    import traceback
    traceback.print_exc()