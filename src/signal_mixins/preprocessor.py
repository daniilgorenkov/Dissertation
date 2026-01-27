import os
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))
# print(sys.path)
from config import SignalPreprocessorConfig, Paths
import configparser
import numpy as np
import re

# import pandas as pd
from scipy.signal import find_peaks
from mixins.file_operator import FileOperator


class SignalPreprocessor(FileOperator):

    def __init__(self):
        super().__init__()
        self.cfg_parser = configparser.ConfigParser(allow_no_value=True)

    def _build_files(self):

        # Получаем все имена папок и сортируем по числу в названии
        signal_dirs = sorted(
            os.listdir(Paths._SIGNALS), key=lambda x: int(x.replace("signal", "")) if x.startswith("signal") else -1
        )

        # Берём только элементы с чётным индексом (0, 2, 4...) → это 1-я, 3-я, 5-я...
        _even_paths = [os.path.join(Paths._SIGNALS, folder_name) for i, folder_name in enumerate(signal_dirs) if i % 2 == 0] # fmt:skip
        _odd_paths = [os.path.join(Paths._SIGNALS, folder_name) for i, folder_name in enumerate(signal_dirs) if i % 2 == 1] # fmt:skip

        def get_valid_files(paths):
            valid_files = []
            for p in paths:
                for filename in os.listdir(p):
                    if filename.endswith(SignalPreprocessorConfig.ALLOWED_EXTENSIONS):
                        valid_files.append(os.path.join(p, filename))
            return valid_files

        return get_valid_files(_even_paths), get_valid_files(_odd_paths)

    def filter_fpaths_by_sequences(self, paths: str):
        filtered_fnames = []
        for p in paths:
            for seq_name in SignalPreprocessorConfig.SEQUENCES:
                if os.path.basename(p).startswith(seq_name):
                    filtered_fnames.append(p)
        return filtered_fnames

    def get_forces_trial_results_paths(self, fpaths: list, vertical: bool = True):
        vertical_forces = []
        side_forces = []

        for f in fpaths:
            parts = os.path.basename(f).split("_")
            indicator = int(parts[2].split(".")[0])
            if indicator == 1 and vertical:
                vertical_forces.append(f)
            elif indicator == 2 and not vertical:
                side_forces.append(f)

        if vertical:
            return vertical_forces
        else:
            return side_forces

    def get_trial_paths(self) -> tuple[list, list]:
        """Returns:
        vertical_even, side_even, vertical_odd, side_odd"""
        even_paths, odd_paths = self._build_files()
        even_paths = self.filter_fpaths_by_sequences(even_paths)
        odd_paths = self.filter_fpaths_by_sequences(odd_paths)

        vertical_even = self.get_forces_trial_results_paths(even_paths)
        side_even = self.get_forces_trial_results_paths(even_paths, vertical=False)

        vertical_odd = self.get_forces_trial_results_paths(odd_paths)
        side_odd = self.get_forces_trial_results_paths(odd_paths, vertical=False)
        return vertical_even, side_even, vertical_odd, side_odd

    def get_mera_params(self, mera_file, channel):
        self.cfg_parser.read(mera_file, encoding="windows-1251")

        k1 = float(self.cfg_parser[channel].get("k1", 1.0))
        k0 = float(self.cfg_parser[channel].get("k0", 0.0))
        units = self.cfg_parser[channel]["yunits"]
        return {"k1": k1, "k0": k0, "units": units}

    def read_signal(self, fpath):
        return np.fromfile(fpath, dtype=np.float32)

    def convert_to_forces(self, signal, k0, k1):
        return signal * k1 + k0

    def find_wheel_impacts(self, signal, vertical: bool = True, even: bool = True):
        height_value = 13.5 if vertical else 0.7
        peaks, _ = find_peaks(signal, height=height_value, distance=500)
        return peaks[6:10] if even else peaks[10:15]

    def get_channel_name(self, fpath: str) -> str:
        return os.path.splitext(os.path.basename(fpath))[0]

    def find_mera_for_signal(self, fpath):
        folder = os.path.dirname(fpath)
        folder_name = os.path.basename(folder)
        mera_path = os.path.join(folder, folder_name + ".mera")

        if not os.path.exists(mera_path):
            raise FileNotFoundError(f".mera не найден: {mera_path}")

        return mera_path

    def get_group_for_signal(self, num: int):
        """Возвращает группу для номера сигнала или None"""
        for group in SignalPreprocessorConfig.GROUP_RANGES:
            if group["start"] <= num <= group["end"]:
                return group
        return None

    def get_signal_number_from_path(self, file_path: str) -> int:
        """Извлекает номер сигнала из пути: signal0580 → 580"""
        match = re.search(r"signal(\d{4})", file_path)
        return int(match.group(1)) if match else -1

    def concat_trials_grouped(self, trials: dict, vertical: bool = True):
        """
        Возвращает список групп, где каждая группа содержит:
        - v: значение скорости (20, 40, 60...)
        - start_num, end_num: диапазон сигналов
        - peaks: объединённый список всех пиков в группе
        """
        force = "vertical" if vertical else "side"
        if force not in trials:
            return []

        # Собираем и сортируем по номеру сигнала
        items = sorted(trials[force], key=lambda x: self.get_signal_number_from_path(x["file"]))

        # Группируем по диапазонам
        groups = []
        current_group = None
        current_peaks = []

        for item in items:
            num = self.get_signal_number_from_path(item["file"])
            if num < 0:
                continue

            group_info = self.get_group_for_signal(num)
            if not group_info:
                # Если сигнал не попал ни в одну группу — можно логировать или пропустить
                continue

            # Новая группа?
            if current_group is None or current_group["v"] != group_info["v"]:
                # Сохраняем предыдущую, если была
                if current_peaks:
                    current_group["peaks"] = current_peaks
                    groups.append(current_group)

                # Начинаем новую
                current_group = {
                    "v": group_info["v"],
                    "start_num": group_info["start"],
                    "end_num": group_info["end"],
                    "peaks": [],  # пока пустой, заполним в конце
                }
                current_peaks = []

            # Добавляем пики
            peaks = item["peaks_values"]
            if len(peaks) > 0:
                current_peaks.extend(peaks.tolist())

        # Не забываем последнюю группу
        if current_peaks and current_group:
            current_group["peaks"] = current_peaks
            groups.append(current_group)

        return groups

    def preprocess_all_signals(self):
        vertical_even, side_even, vertical_odd, side_odd = self.get_trial_paths()

        paths = {
            "vertical": {
                "even": vertical_even,
                "odd": vertical_odd,
            },
            "side": {
                "even": side_even,
                "odd": side_odd,
            },
        }

        results = {
            "vertical": {"even": [], "odd": []},
            "side": {"even": [], "odd": []},
        }

        for group_name, parity_groups in paths.items():
            for parity, fpaths in parity_groups.items():
                for fpath in fpaths:

                    # --- обрабатываем только бинарные файлы
                    if not fpath.endswith(".dat"):
                        continue

                    channel = self.get_channel_name(fpath)
                    mera_file = self.find_mera_for_signal(fpath)
                    params = self.get_mera_params(mera_file, channel)
                    signal = self.read_signal(fpath)
                    forces = self.convert_to_forces(signal, params["k0"], params["k1"])

                    even = True if parity == "even" else False

                    # --- ищем удары колёс
                    if group_name == "vertical":
                        max_threshold = SignalPreprocessorConfig.VERTICAL_FORCE_PEAK_THRESHOLD
                        min_threshold = 23.0
                        wheel_peaks = self.find_wheel_impacts(forces, even=even)
                    else:
                        max_threshold = SignalPreprocessorConfig.SIDE_FORCE_PEAK_THRESHOLD
                        min_threshold = -max_threshold
                        wheel_peaks = self.find_wheel_impacts(forces, False, even=even)

                    results[group_name][parity].append(
                        {
                            "file": fpath,
                            "channel": channel,
                            "mera": mera_file,
                            "units": params["units"],
                            "peaks_values": np.clip(forces[wheel_peaks], min_threshold, max_threshold),
                        }
                    )

        self.save(results, "trial_forces")

        vertical_trials = self.concat_trials_grouped(
            {"vertical": results["vertical"]["even"] + results["vertical"]["odd"]}
        )

        side_trials = self.concat_trials_grouped(
            {"side": results["side"]["even"] + results["side"]["odd"]}, vertical=False
        )

        self.save(vertical_trials, "vertical_trials")
        self.save(side_trials, "side_trials")


if __name__ == "__main__":
    prep = SignalPreprocessor()
    prep.preprocess_all_signals()
