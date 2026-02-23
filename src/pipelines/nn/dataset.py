import numpy as np
import torch
from torch.utils.data import Dataset


class WheelSignalsDataset(Dataset):
    def __init__(
        self, X: np.ndarray, y_fault: np.ndarray, y_profile: np.ndarray | None = None, augment_shift: bool = True
    ):
        self.X = X.astype(np.float32)
        self.y_fault = y_fault.astype(np.int64)
        self.y_profile = None if y_profile is None else y_profile.astype(np.int64)
        self.augment_shift = augment_shift

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (2, L)

        if self.augment_shift:
            shift = np.random.randint(0, x.shape[-1])
            x = np.roll(x, shift, axis=-1)

        x = torch.from_numpy(x)

        yf = torch.tensor(self.y_fault[idx], dtype=torch.long)
        if self.y_profile is None:
            return x, yf

        yp = torch.tensor(self.y_profile[idx], dtype=torch.long)
        return x, yf, yp

if __name__ == "__main__":