from dataset import WheelSignalsDataset
from torch import nn

class SignalEncoder(nn.Module):
    def __init__(self, n_points=1536, patch=32, d_model=128, dropout=0.1):
        super().__init__()
        assert n_points % patch == 0
        self.proj = nn.Conv1d(2, d_model, kernel_size=patch, stride=patch, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):           # x: (B, 2, N)
        t = self.proj(x)            # (B, d_model, T)
        t = t.transpose(1, 2)       # (B, T, d_model)
        return self.drop(self.norm(t))
    
if __name__ == "__main__":
    