import os
import cv2
import torch
from model_arch import ViolenceDetectionModel
from dataset import extract_frames

def run_inference(video_path, model_path, output_name='result.mp4'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ViolenceDetectionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    
    with torch.no_grad():
        frames_input = extract_frames(video_path)
        frames_tensor = torch.FloatTensor(frames_input).unsqueeze(0).to(device)
        output = model(frames_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    label = "VIOLENCE DETECTED" if predicted.item() == 1 else "NORMAL ACTIVITY"
    color = (0, 0, 255) if predicted.item() == 1 else (0, 255, 0)
    
    print(f"🎬 Processing: {os.path.basename(video_path)}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"{label} ({confidence.item()*100:.2f}%)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"✅ Saved to: {output_name}")

if __name__ == "__main__":
    run_inference('test_video.avi', 'models/violence_model.pth')