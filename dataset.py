# dataset.py
import os
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import unicodedata

def get_transform(train=True):
    transforms = []
    if train:
         transforms += [
              T.RandomRotation(2, expand=False),
              T.ColorJitter(brightness=0.15, contrast=0.15),
              T.RandomAffine(0, translate=(0.02,0.02), shear=1),
              ]
         
    transforms += [
                    T.PILToTensor(),                # returns uint8 tensor [C,H,W]
                    T.ConvertImageDtype(torch.float32),
                    T.Normalize((0.5,), (0.5,))
                ]
    return T.Compose(transforms)


class OCRDataset(Dataset):
    def __init__(self, img_folder: str, labels_file: str, img_h: int = 32, img_w: int = 256, train: bool = True):
        self.img_folder = img_folder
        self.img_h = img_h
        self.img_w = img_w
        self.samples: List[Tuple[str,str]] = []
        with open(labels_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 1:
                    img_name = parts[0]
                    label = ""
                else:
                    img_name, label = parts
                self.samples.append((img_name, label))
        self.transform = get_transform(train)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_folder, os.path.basename(img_name))
        img = Image.open(img_path).convert("L")
        # preserve aspect ratio, resize height to img_h
        w, h = img.size
        new_w = max(1, int(self.img_h * w / h))
        if new_w > self.img_w:
            new_w = self.img_w
        img = img.resize((new_w, self.img_h), Image.BILINEAR)
        # pad to fixed width (use white background 255)
        padded = Image.new("L", (self.img_w, self.img_h), 255)
        padded.paste(img, (0, 0))
        if self.transform:
            padded = self.transform(padded)
        return padded, label
    
# collate function for variable widths — pads to max width in batch; returns widths
def collate_fn(batch):
      imgs, labels = zip(*batch)
      # imgs are tensors [C,H,W]
      max_w = max(img.shape[-1] for img in imgs)
      padded = []
      widths = []
      for img in imgs:
          c, h, w = img.shape
          pad = torch.nn.functional.pad(img, (0, max_w - w, 0, 0))
          padded.append(pad)
          widths.append(w)
      imgs_tensor = torch.stack(padded, 0)
      return imgs_tensor, list(labels), torch.tensor(widths, dtype=torch.long)