import os
import cv2
from ultralytics import YOLO

# Path to the trained model
model_path = os.path.join('runs', 'train', 'exp', 'weights', 'best.pt')

# Function to train the model
def train_model():
    model = YOLO('yolov8n.pt')  # Load a pre-trained model
    # Get the path of the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Set the relative path for data.yaml
    data_path = os.path.join(current_dir, 'dataset', 'data.yaml')  # Update path here
    model.train(data=data_path, epochs=50, imgsz=640)  # Adjust as needed
    print("Training completed. Check 'runs/train/exp/weights/' for the best.pt file.")

# Function for inference
def run_inference():
    # Load the trained YOLOv8 model
    model = YOLO(model_path)

    def value():
        val = input("Enter file name or press enter to start webcam: \n")
        if val == "":
            val = 0
        return val

    # For video capture
    cap = cv2.VideoCapture(value())

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to read a frame from the video source.")
            break

        # Perform inference
        results = model(img)

        # Process results
        for result in results:
            for bbox in result.boxes.xyxy:  # Get bounding boxes
                x1, y1, x2, y2, conf, cls = bbox
                if conf > 0.5:  # Confidence threshold
                    label = model.names[int(cls)]  # Get class name

                    # Draw bounding box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    if not os.path.exists(model_path):
        print("Model not found. Starting training...")
        train_model()
    else:
        print("Model found. Starting inference...")
        run_inference()
