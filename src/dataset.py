import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def extract_frames(video_path, seq_len=16, img_size=224):
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(total_frames / seq_len), 1)
    
    frames = []
    for i in range(seq_len):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames_window)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame / 255.0) 
    cap.release()
    
    while len(frames) < seq_len:
        frames.append(np.zeros((img_size, img_size, 3)))
        
    return np.array(frames).transpose(0, 3, 1, 2) # (Seq, Channel, H, W)

class ViolenceDataset(Dataset):
    
    def __init__(self, video_paths, labels, seq_len=16):
        self.video_paths = video_paths
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = extract_frames(video_path, self.seq_len)
        return torch.FloatTensor(frames), torch.tensor(label, dtype=torch.long)