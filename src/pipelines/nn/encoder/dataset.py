import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset

import numpy as np
import torch
from torch.utils.data import Dataset


class WheelSignalsDataset(Dataset):
    def __init__(self, data: list[dict], augment_shift: bool = True, return_filename: bool = False):
        self.data = []
        self.augment_shift = augment_shift
        self.return_filename = return_filename

        for i, item in enumerate(data):
            x = np.asarray(item["X"], dtype=np.float32)

            if x.ndim != 2:
                raise ValueError(f"Expected X with shape (2, L), got {x.shape} at item {i}")

            if x.shape[0] != 2:
                raise ValueError(f"Expected 2 channels, got {x.shape} at item {i}")

            self.data.append(
                {
                    "X": x,
                    "y_fault": int(item["y_fault"]),
                    "y_profile": int(item["y_profile"]),
                    "filename": item.get("filename", None),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        x = item["X"]  # (2, L)

        if self.augment_shift:
            shift = np.random.randint(0, x.shape[-1])
            x = np.roll(x, shift, axis=-1)

        x = torch.tensor(x, dtype=torch.float32)
        y_fault = torch.tensor(item["y_fault"], dtype=torch.long)
        y_profile = torch.tensor(item["y_profile"], dtype=torch.long)

        if self.return_filename:
            return x, y_fault, y_profile, item["filename"]

        return {"X": x, "y_fault": y_fault, "y_profile": y_profile}
