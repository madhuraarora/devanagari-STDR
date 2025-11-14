# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
      
      def __init__(self, img_h: int, num_channels: int, num_classes: int, hidden_size: int = 256, img_w_test=256):
          super().__init__()
          self.cnn = nn.Sequential(
              nn.Conv2d(num_channels, 64, 3, 1, 1),
              nn.ReLU(True),
              nn.MaxPool2d(2,2),
              nn.Conv2d(64, 128, 3, 1, 1),
              nn.ReLU(True),
              nn.MaxPool2d(2,2),
              nn.Conv2d(128, 256, 3, 1, 1),
              nn.ReLU(True),
              nn.Conv2d(256, 256, 3,1,1),
              nn.ReLU(True),
              nn.MaxPool2d((2,1),(2,1)),
              nn.Conv2d(256, 512, 3,1,1),
              nn.ReLU(True),
              nn.BatchNorm2d(512),
              nn.Conv2d(512, 512, 3,1,1),
              nn.ReLU(True),
              nn.BatchNorm2d(512),
              nn.MaxPool2d((2,1),(2,1)),
              nn.Conv2d(512, 512, 2,1,0),
              nn.ReLU(True)
          )

          # sanity check: run dummy through cnn to see final height
          with torch.no_grad():
              tmp = torch.zeros(1, num_channels, img_h, min(64, img_w_test))
              out = self.cnn(tmp)
              _, _, h_out, _ = out.size()
              if h_out != 1:
                  # warn; we'll handle in forward via adaptive pooling
                  # but it's good to know at init
                  print(f"[CRNN init] conv output height is {h_out}; adaptivepooling will be used in forward if needed")
          self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
          self.embedding = nn.Linear(hidden_size*2, num_classes)
      
      def forward(self, x):
          # x: [B, C, H, W]
          conv = self.cnn(x)
          b, c, h, w = conv.size()
          if h != 1:
              
              conv = F.adaptive_avg_pool2d(conv, (1, w))
          conv = conv.squeeze(2)  # [B, C, W]
          conv = conv.permute(0, 2, 1)  # [B, W, C]
          rnn_out, _ = self.rnn(conv)  # batch_first=True
          out = self.embedding(rnn_out)  # [B, W, num_classes]
          out = out.permute(1, 0, 2)  # [W, B, num_classes] for CTC
          return out
