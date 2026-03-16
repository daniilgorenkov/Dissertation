import numpy as np


class DataProcessor:

    @staticmethod
    def extract_scalar_label(y) -> int:
        y = np.asarray(y)
        if y.size == 0:
            raise ValueError(f"Empty label: {y}")
        return int(y.reshape(-1)[0])

    def pad_or_crop_window(self, x: np.ndarray, target_len: int = 1536) -> np.ndarray:
        """Convert (2, L) -> (2, target_len) via crop or right zero-pad."""
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 2:
            raise ValueError(f"Expected window shape (2, L), got {x.shape}")
        if x.shape[0] != 2:
            raise ValueError(f"Expected 2 channels, got shape {x.shape}")

        length = x.shape[1]
        if length == target_len:
            return x
        if length > target_len:
            return x[:, :target_len]
        return np.pad(x, ((0, 0), (0, target_len - length)), mode="constant").astype(np.float32)

    def flatten_windowed_data(
        self,
        data: list[dict],
        target_len: int = 1536,
        drop_empty: bool = True,
    ) -> list[dict]:
        flat_data = []
        append = flat_data.append

        for item_idx, item in enumerate(data):
            X = np.asarray(item["X"], dtype=np.float32)

            if X.ndim != 3:
                raise ValueError(f"Expected X shape (n_windows, 2, L), got {X.shape} in item {item_idx}")

            if X.shape[0] == 0:
                if drop_empty:
                    continue
                raise ValueError(f"Empty X in item {item_idx}, filename={item.get('filename')}")

            y_fault = self.extract_scalar_label(item["y_fault"])
            y_profile = self.extract_scalar_label(item["y_profile"])
            filename = item.get("filename")

            for window in X:
                append(
                    {
                        "X": self.pad_or_crop_window(window, target_len=target_len),
                        "y_fault": y_fault,
                        "y_profile": y_profile,
                        "filename": filename,
                    }
                )

        return flat_data
