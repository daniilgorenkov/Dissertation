from torch import nn


class SignalEncoder(nn.Module):
    def __init__(self, n_points=1536, patch=32, d_model=128, dropout=0.1):
        super().__init__()
        assert n_points % patch == 0
        self.proj = nn.Conv1d(2, d_model, kernel_size=patch, stride=patch, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, 2, N)
        t = self.proj(x)  # (B, d_model, T)
        t = t.transpose(1, 2)  # (B, T, d_model)
        return self.drop(self.norm(t))


class WheelModel(nn.Module):
    def __init__(self, n_points=1536, patch=32, d_model=128, dropout=0.1):
        super().__init__()

        self.encoder = SignalEncoder(n_points, patch, d_model, dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fault_head = nn.Linear(d_model, 2)  # исправное/неисправное
        self.profile_head = nn.Linear(d_model, 3)  # 0 - новое, 1 - изношенное, 2 - критическое

    def forward(self, x):

        t = self.encoder(x)  # (B, T, 128)

        t = t.transpose(1, 2)  # (B, 128, T)
        t = self.pool(t).squeeze(-1)  # (B,128)

        fault = self.fault_head(t)
        profile = self.profile_head(t)

        return fault, profile
