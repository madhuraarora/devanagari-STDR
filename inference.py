# inference.py
import torch

def greedy_decode(logits, charset):
      # logits: [W, B, C]
      preds = logits.argmax(2).transpose(0,1)  # [B, W]
      results = []
      blank_idx = len(charset) - 1
      for seq in preds:
            prev = -1
            out = []
            
            for idx in seq.cpu().numpy():
              if idx != prev and idx < blank_idx:
                  out.append(charset[idx])
              prev = idx
            results.append("".join(out))
      return results


def load_charset(path="master_charset.txt"):
      with open(path, "r", encoding="utf-8") as f:
          return [c.rstrip('\n') for c in f.readlines() if c.rstrip('\n')]
      

def predict_image(model, img_tensor, charset, device="cpu"):
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)
        logits = model(img_tensor)
        text = greedy_decode(logits, charset)[0]
    return text