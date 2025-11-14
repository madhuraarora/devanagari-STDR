# basic_training.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OCRDataset, collate_fn
from model import CRNN
from token_utils import build_charset_from_label_files
from training_common import encode_label_graphemes, cer, grapheme_split
from tqdm import tqdm
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_or_build_charset(labels_files, save_path="master_charset.txt"):
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as fh:
            charset = [l.rstrip("\n") for l in fh.readlines() if l.rstrip("\n")]
    else:
        charset = build_charset_from_label_files(labels_files, save_path=save_path)
    return charset

def greedy_decode_logits_to_strings(logits, charset):
    preds = logits.argmax(2).transpose(0,1)  # [B, T]
    blank_idx = len(charset) - 1
    results = []
    for seq in preds:
        prev = -1
        out = []
        for idx in seq.cpu().numpy():
            if idx != prev and idx < blank_idx:
                out.append(charset[idx])
            prev = idx
        results.append("".join(out))
    return results

def main():
    img_folder = "images/train_basic/basic_syn1"
    labels_file = os.path.join(img_folder, "basic_syn1.txt")
    labels_files = [labels_file]

    charset = load_or_build_charset(labels_files, save_path="master_charset.txt")
    charset_map = {g: i for i, g in enumerate(charset)}

    dataset = OCRDataset(img_folder, labels_file, img_h=32, img_w=256, train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = CRNN(img_h=32, num_channels=1, num_classes=len(charset)).to(device)
    checkpoint_path = "checkpoints/basic_best.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded existing weights from {checkpoint_path}")

    criterion = nn.CTCLoss(blank=len(charset)-1, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    best_cer = float('inf')

    for epoch in range(1, 9):
        model.train()
        total_loss = 0.0
        total_cer = 0.0
        seen_chars = 0
        skipped_samples_total = 0

        for imgs, labels, widths in tqdm(loader, desc=f"[basic] Epoch {epoch}"):
            # encode targets first and decide which samples to keep
            encoded_per_sample = [encode_label_graphemes(l, charset_map) for l in labels]
            keep_idx = [i for i, enc in enumerate(encoded_per_sample) if enc.numel() > 0]
            skipped_samples_total += (len(labels) - len(keep_idx))
            if len(keep_idx) == 0:
                continue

            # select only kept images and labels
            imgs_keep = imgs[keep_idx].to(device)          # [B_keep, C, H, W]
            labels_keep = [labels[i] for i in keep_idx]
            targets_list = [encoded_per_sample[i] for i in keep_idx]
            targets = torch.cat(targets_list).to(device)
            target_lengths = torch.tensor([t.numel() for t in targets_list], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(imgs_keep)               # [T, B_keep, C]
            log_probs = F.log_softmax(logits, dim=2)
            input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long).to(device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

            # quick sampled CER calculation (char-weighted)
            with torch.no_grad():
                preds = greedy_decode_logits_to_strings(logits.cpu(), charset)
                for pred_str, tgt_str in zip(preds, labels_keep):
                    tgt_norm = "".join(grapheme_split(tgt_str))
                    pred_norm = "".join(grapheme_split(pred_str))
                    total_cer += cer(pred_norm, tgt_norm) * max(1, len(tgt_norm))
                    seen_chars += max(1, len(tgt_norm))

        avg_loss = total_loss / max(1, len(loader))
        avg_cer = (total_cer / seen_chars) if seen_chars > 0 else 0.0
        print(f"[basic] Epoch {epoch} avg loss: {avg_loss:.4f} avg CER (char-weighted): {avg_cer:.4f} skipped_samples_this_epoch: {skipped_samples_total}")
                # --- show sample predictions ---
        model.eval()
        import random
        from inference import greedy_decode

        with torch.no_grad():
            # take 3 random samples from the dataset
            sample_idxs = random.sample(range(len(dataset)), min(3, len(dataset)))
            for i in sample_idxs:
                img, gt = dataset[i]
                img = img.unsqueeze(0).to(device)
                logits = model(img)
                preds = greedy_decode(logits, charset)
                pred_text = preds[0]
                print(f"[Sample] GT: {gt}  |  Pred: {pred_text}")
        print("-" * 80)


        if avg_cer < best_cer:
            best_cer = avg_cer
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/basic_best.pth")
            print("Saved checkpoints/basic_best.pth")

if __name__ == "__main__":
    main()
