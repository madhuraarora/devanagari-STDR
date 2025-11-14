import os
import torch
from torch.utils.data import DataLoader
from dataset import OCRDataset, collate_fn
from model import CRNN
from token_utils import build_charset_from_label_files
from training_common import encode_label_graphemes, cer, grapheme_split
from inference import greedy_decode

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_or_build_charset(labels_files, save_path="master_charset.txt"):
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as fh:
            charset = [l.rstrip("\n") for l in fh.readlines() if l.rstrip("\n")]
    else:
        charset = build_charset_from_label_files(labels_files, save_path=save_path)
    return charset


def wer(prediction: str, target: str) -> float:
    """Compute Word Error Rate (WER) using edit distance."""
    pred_words = prediction.split()
    tgt_words = target.split()
    dp = [[0] * (len(tgt_words) + 1) for _ in range(len(pred_words) + 1)]

    for i in range(len(pred_words) + 1):
        dp[i][0] = i
    for j in range(len(tgt_words) + 1):
        dp[0][j] = j

    for i in range(1, len(pred_words) + 1):
        for j in range(1, len(tgt_words) + 1):
            if pred_words[i - 1] == tgt_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[-1][-1] / max(1, len(tgt_words))


@torch.no_grad()
def evaluate(model, dataloader, charset):
    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    seen_chars = 0
    seen_sentences = 0

    for imgs, labels, _ in dataloader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = greedy_decode(logits, charset)

        for pred_str, tgt_str in zip(preds, labels):
            tgt_norm = "".join(grapheme_split(tgt_str))
            pred_norm = "".join(grapheme_split(pred_str))
            sample_cer = cer(pred_norm, tgt_norm)
            sample_wer = wer(pred_norm, tgt_norm)

            total_cer += sample_cer * max(1, len(tgt_norm))
            seen_chars += max(1, len(tgt_norm))
            total_wer += sample_wer
            seen_sentences += 1

    avg_cer = total_cer / seen_chars
    avg_wer = total_wer / seen_sentences

    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")


def main():
    img_folder = "images/test_crop"
    labels_file = "images/test_crop/test_crop.txt"
    labels_files = [labels_file]

    charset = load_or_build_charset(labels_files, save_path="master_charset.txt")

    dataset = OCRDataset(img_folder, labels_file, img_h=32, img_w=256, train=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = CRNN(img_h=32, num_channels=1, num_classes=len(charset)).to(device)

    ckpt = "checkpoints/phase2_best.pth"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        raise FileNotFoundError(f"{ckpt} not found!")

    evaluate(model, loader, charset)


if __name__ == "__main__":
    main()
