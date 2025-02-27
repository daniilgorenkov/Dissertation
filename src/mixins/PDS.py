import os
from typing import Generator
from config import Paths, SimulationNames, SimulationSpeeds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
from scipy.stats import skew, kurtosis
from sklearn.metrics import precision_score, recall_score, f1_score

# import catboost
import sklearn
from sklearn.metrics import classification_report
import datetime


class PDS:
    """Pretrained Diagnostic System"""

    def __init__(self, force: str = "vertical"):
        self.PROFILES = [SimulationNames.GREB_24, SimulationNames.GREB_26, SimulationNames.GREB_30, SimulationNames.GREB_GOST, SimulationNames.GREB_UM]  # fmt:skip
        self.WAY_TYPES = [SimulationNames.STRAIGHT, SimulationNames.CURVE_350, SimulationNames.CURVE_650]  # fmt:skip
        self.WAGON_CONFIGS = [SimulationNames.EMPTY, SimulationNames.LOADED]  # fmt:skip
        self.FAULTS = [
            SimulationNames.NO_FAULT,
            SimulationNames.POLZUN,
            SimulationNames.ELLIPS,
        ]
        self.FORCE = force
        self.plot_encoder = {
            "loaded": "Груженый",
            "empty": "Порожний",
            "straight": "прямая",
            "curve_350": "кривая 350 м",
            "curve_650": "кривая 650 м",
            "normal": "без неисправностей",
            "polzun15": "ползун",
            "ellips10": "неравномерный прокат",
        }

    def get_result(
        self,
        wagon_config: str,
        way_type: str,
        fault: str,
        speed: int,
        profile: str = "gost",
    ) -> pd.DataFrame:
        """
        Result of the simulation
        """

        fname = "_".join([wagon_config, way_type, fault, str(speed), profile])
        fname_ext = fname + ".csv"
        FULL_PATH = os.path.join(
            Paths.SIMULATION_RESULTS, self.FORCE, wagon_config, fname_ext
        )

        file = pd.read_csv(FULL_PATH, encoding="latin-1")
        COL_NAMES = ["time_step", fname]
        file.columns = COL_NAMES
        file.set_index("time_step", inplace=True)
        return file

    def get_profile_results(
        self, wagon_config: str, way_type: str, fault: str, speed: int
    ) -> Generator:
        """Simulation results by profiles"""
        for profile in self.PROFILES:
            yield self.get_result(wagon_config, way_type, fault, speed, profile)

    def plot_profile_results(
        self,
        wagon_config: str,
        way_type: str,
        fault: str,
        speed: int,
        xlim: tuple[int, int] = None,
        ylim: tuple[int, int] = None,
    ):
        """Plot simulation results by profiles"""

        files = self.get_profile_results(wagon_config, way_type, fault, speed)
        FILE = pd.concat(files, axis=1)
        FILE.columns = self.PROFILES
        plt.figure(figsize=(12, 8))
        plt.grid(True)
        sns.lineplot(FILE)
        plt.title(
            f"{self.plot_encoder[wagon_config]} вагон, {self.plot_encoder[way_type]}, {self.plot_encoder[fault]}, скорость {speed} км/ч "
        )
        plt.xlabel("Время, с")
        if self.FORCE == "vertical":
            plt.ylabel("Вертикальная сила, Н")
        elif self.FORCE == "side":
            plt.ylabel("Боковая сила, Н")

        if xlim and ylim:
            plt.xlim(xlim)
            plt.ylim(ylim)
        plt.show()

    def get_speed_results(
        self,
        wagon_config: str,
        way_type: str,
        fault: str,
        profile: str = "gost",
    ) -> Generator:
        """Получение расчета сразу по всем скоростям"""
        SPEED = (
            SimulationSpeeds.STRAIGHT
            if "straight" in way_type
            else SimulationSpeeds.CURVE
        )

        for v in SPEED:
            file = self.get_result(wagon_config, way_type, fault, v, profile)
            yield file

    def get_full_calculations(self) -> dict:
        """Получение словаря со всеми расчетами"""
        simulation_results = {}

        for wagon in self.WAGON_CONFIGS:
            simulation_results[wagon] = {}
            for way in self.WAY_TYPES:
                simulation_results[wagon][way] = {}
                for fault in self.FAULTS:
                    simulation_results[wagon][way][fault] = {}
                    for wheel in self.PROFILES:
                        clear_output(True)
                        print(f"{wagon}\n{way}\n{fault}\n{wheel}\n------")
                        all_speed_results = self.get_speed_results(wagon, way, fault, wheel)  # fmt:skip
                        simulation_results[wagon][way][fault][wheel] = pd.concat([speed_result for speed_result in all_speed_results], axis=1)  # fmt:skip

        return simulation_results

    def time_split(self, v: int) -> int:
        """Определение временного промежутка полного оборота колеса
        1. v - скорость движения поезда, км/ч
        """
        speed = v / 3.6
        lenght = 2 * np.pi * 0.475
        t = lenght / speed
        return t

    def get_time_splits(self, data: pd.DataFrame) -> list:
        """Возвращает индексы по которым нужно производить обрез
        Индексы высчтитываются в зависимости от скорости движения вагона и радиуса колеса
        """

        res = []

        start_point = 1.4

        time_max_point = data.index.max()  # Максимальное время

        if data.columns[0].split("_")[1] == "straight":
            col_name = data.columns[0].split("_")  # 10,20,30... км/ч
            speed = int(col_name[3])
            wheel_rotate_num = self.time_split(speed)  # 1.007 сек

        elif data.columns[0].split("_")[1] == "curve":
            col_name = data.columns[0].split("_")  # 10,20,30... км/ч
            speed = int(col_name[4])
            wheel_rotate_num = self.time_split(speed)  # 1.007 сек

        num_folds = (
            time_max_point - 1
        ) // wheel_rotate_num  # Сколько всего фолдов получится сделать

        res.append(start_point)

        for _ in range(int(num_folds)):
            start_point += wheel_rotate_num
            res.append(start_point)

        return res

    def time_indexes(self, frames: list[pd.DataFrame]) -> dict:
        """Возвращает словарь индексов по которым нужно производить обрез, где
        индексы высчтитываются в зависимости от скорости движения вагона и радиуса колеса
        """
        res = {}

        for n in range(len(frames)):
            str = frames[n].columns[0].split("_")
            for s in str:
                if s.isdigit():  # тут нужно придумать исключение для толщины гребней
                    if int(s) != 350 and int(s) != 650 and int(s) != 24:
                        name = s
                        res[name] = self.get_time_splits(frames[n])
        return res

    def get_all_time_indexes(
        self,
        calculations: dict[dict[dict[dict[list[pd.DataFrame]]]]],
    ) -> dict:
        """Получение всех индексов времени по которым надо делить расчеты в виде словаря"""

        time_idxs = {}
        for wagon in self.WAGON_CONFIGS:
            time_idxs[wagon] = {}
            for way in self.WAY_TYPES:
                time_idxs[wagon][way] = {}
                for fault in self.FAULTS:
                    time_idxs[wagon][way][fault] = {}
                    for wheel in self.PROFILES:
                        clear_output(True)
                        print(f"{wagon}\n{way}\n{fault}\n{wheel}\n------")
                        time_idxs[wagon][way][fault][wheel] = self.time_indexes(
                            calculations[wagon][way][fault][wheel]
                        )
        return time_idxs

    def get_splitted_dataframe(data: pd.DataFrame, indexes: list) -> pd.DataFrame:
        """Разделение одного результата расчета на несколько других по полному обороту колеса
        1. `data` - датафрейм с расчетом
        2. `indexes` - индексы по которым нужно делить расчет"""

        zeros = np.zeros((214, 1))
        common_df = pd.DataFrame(zeros)

        for i in range(len(indexes)):
            if i < len(indexes) - 1:
                seq = data[
                    (data.index >= indexes[i]) & (data.index <= indexes[i + 1])
                ]  # срез по точкам
            else:
                seq = data[data.index > indexes[i]]
            common_df = pd.concat([common_df, seq], axis=1)

        df = common_df.drop(0, axis=1)
        num_cols = len(df.columns)

        df.columns = [
            [data.columns[0] for i in range(num_cols)],
            [i for i in range(num_cols)],
        ]

        return df

    def get_skew_kurt(data: pd.DataFrame) -> pd.DataFrame:
        """Получение дополнительных фичей для расчетов"""

        cols = data.columns

        skews = []
        kurtosises = []

        for i in cols:
            skew_ = skew(data[i].dropna().to_numpy())
            kurt_ = kurtosis(data[i].dropna().to_numpy())
            skews.append(skew_)
            kurtosises.append(kurt_)

        return pd.DataFrame({"skew": skews, "kurt": kurtosises}, index=cols).T

    def get_description(data: pd.DataFrame) -> pd.DataFrame:
        """Получаем описанный фрейм и к нему добавляем доп фичи"""

        summ = data.sum()
        variance = data.var()
        skew_kurt = PDS.get_skew_kurt(data)
        desc = data.describe()

        summ_var = pd.concat([variance, summ], axis=1).T
        summ_var.index = ["var", "sum"]

        df = pd.concat([desc, summ_var, skew_kurt], axis=0)
        return df

    def make_frame_from_splits(calculations: dict, time_indexes: dict) -> pd.DataFrame:
        """Объединение всех разделенных расчетов на фолды и создание фичей"""

        wagon_cfg = calculations.keys()
        way_cfg = calculations["empty"].keys()
        fault_cfg = calculations["empty"]["straight"].keys()
        wheel_cfg = calculations["empty"]["straight"]["normal"].keys()
        speed_cfg = time_indexes["empty"]["straight"]["normal"]["gost"].keys()
        lenght = len(calculations["empty"]["straight"]["normal"]["gost"])

        zeros = np.zeros((1, 12))
        common_df = pd.DataFrame(zeros)

        n = 0

        for wagon in wagon_cfg:
            for way in way_cfg:
                for fault in fault_cfg:
                    for wheel in wheel_cfg:
                        for l, speed in zip(range(lenght), speed_cfg):

                            if "curve" in way:
                                if l > 7 and int(speed) > 80:
                                    continue

                            splitted_df = PDS.get_splitted_dataframe(
                                calculations[wagon][way][fault][wheel][l],
                                time_indexes[wagon][way][fault][wheel][speed],
                            )

                            feats = PDS.get_description(splitted_df)

                            common_df = pd.concat([common_df, feats], axis=1)

                            clear_output(wait=True)
                            print(f"Сделано: {n}")
                            n += 1

        df = common_df.drop(0, axis=0).drop(0, axis=1)

        return df

    def make_pretty_df(data: pd.DataFrame, file_name: str, save: bool) -> pd.DataFrame:
        """Убирает пустые колонки и делает красивые названия колонок и сохраняет результат"""

        df_ = data.copy()
        unvalid_cols = [i for i in range(1, 12)]

        df_ = df_.drop(unvalid_cols, axis=1)  # Тут заменил df на df_

        new_cols = pd.MultiIndex.from_tuples(df_.columns)
        df_.columns = new_cols

        if save:
            df_.T.to_parquet(f"{file_name}")

        return df_.T

    def delete_unvalid_cols(data: pd.DataFrame) -> pd.DataFrame:
        """Удаление неликвидных колонок"""
        data_copy = data.T.copy()
        unvalid_cols = []

        for i in data_copy.columns:
            if data_copy[i].nunique() < 3:
                unvalid_cols.append(i)

        df = data_copy.drop(unvalid_cols, axis=1)

        return df.T

    def new_str(value: str) -> str:
        """Замена строки типа `loaded_curve_650_normal_30_greb_30` на `loaded_curve650_normal_30_greb_30`"""

        if "curve" in value and "greb" in value:
            splitted = value.split("_")
            way_cfg = splitted[1]
            curve_m = splitted[2]
            new_word_1 = way_cfg + curve_m
            value = value.replace(curve_m, "")
            value = value.replace(way_cfg + "_", new_word_1)

            greb = splitted[5]
            greb_mm = splitted[6]
            new_word_2 = greb + greb_mm
            value = value.replace(greb_mm, "")
            value = value.replace(greb + "_", new_word_2)

            speed = splitted[4]

            if greb_mm == "30" and speed == "30":
                value = value.split("_")

                if "" in value:
                    value.remove("")
                value.insert(3, "30")
                # print(value)
                value = "_".join(value)

        elif "curve" in value:
            splitted = value.split("_")
            way_cfg = splitted[1]
            curve_m = splitted[2]
            new_word_1 = way_cfg + curve_m
            value = value.replace(curve_m, "")
            value = value.replace(way_cfg + "_", new_word_1)

        return value

    def binarize_target(string: str) -> int:
        """Бинарное кодирование таргета"""

        if (
            string == "normal"
            or string == "newwagonw"
            or string == "gost"
            or string == "greb30"
            or string == "greb28"
        ):
            return 0
        else:
            return 1

    def encode_target(string: str) -> int:
        """Разделение таргета на 3 группы"""

        if string == "normal" or string == "newwagonw" or string == "gost":
            return int(0)

        elif string == "polzun15" or string == "greb28" or string == "greb30":
            return int(1)

        elif string == "ellips10" or string == "greb26" or string == "greb24":
            return int(2)

    def set_tuple_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Установка двухуровневых колонок"""

        df_ = df.copy()
        cols = []

        for i in df_.columns:
            cols.append((f"{prefix}", i))

        multicols = pd.MultiIndex.from_tuples(cols)

        df_.columns = multicols
        return df_

    def save_results(estimator, X_test, y_test, model_type: str, addition: str):

        ave = "micro"
        y_pred = estimator.predict(X_test)
        presicion = precision_score(y_test, y_pred, average=ave)
        recall = recall_score(y_test, y_pred, average=ave)
        f1_ = f1_score(y_test, y_pred, average=ave)

        df = pd.read_csv("stat_results.csv")

        shape = X_test.shape[0]

        line = pd.DataFrame(
            {
                "model": [model_type],
                "step1": [estimator.steps[0][0]],
                "step2": [estimator.steps[1][0]],
                "test_size": [shape],
                "presicion": [presicion],
                "recall": [recall],
                "f1_score": [f1_],
                "addition": [addition],
            }
        )

        updated_stats = (
            pd.concat([df, line], axis=0)
            .drop("Unnamed: 0", axis=1)
            .to_csv("stat_results.csv")
        )

        return updated_stats

    def plot_feature_importance(
        estimator, col_names: list = None, force_side: bool = True
    ):

        plt.rcParams.update({"font.size": 10})

        if type(estimator[1]) == xgboost.sklearn.XGBClassifier:
            coefs = estimator[1].coef_

        elif (
            type(estimator[1]) == catboost.core.CatBoostClassifier
            or lightgbm.sklearn.LGBMClassifier
        ):
            coefs = estimator[1].feature_importances_

        elif type(estimator[1]) == sklearn.linear_model._coordinate_descent.Lasso:
            coefs = estimator[1].coef_

        if coefs.shape[0] == 3 and force_side == False:
            d = {0: "исправного колеса", 1: "ползуна", 2: "неравномерного проката"}

        if coefs.shape[0] == 3 and force_side == True:
            d = {
                0: "исправного гребня",
                1: "средне изношенного гребня",
                2: "сильно изношенного гребня",
            }

        elif coefs.shape[0] == 2:
            d = {0: "исправного вагона", 1: "неисправного вагона"}

        elif len(coefs.shape) == 1:
            d = {0: "модели Catboost"}

        if len(coefs) <= 3:
            for i in range(len(coefs)):
                df = pd.DataFrame(coefs[i]).T
                if col_names:
                    df.columns = col_names
                else:
                    df.columns = estimator[:-1].get_feature_names_out()
                df.index = ["Степень важности"]
                plt.figure().set_size_inches(16, 8)
                plt.title(f"Коэффициенты важности признаков для предсказания {d[i]}")
                sns.barplot(abs(df))
                plt.xticks(rotation=20)
                plt.xticks(fontsize=8)
                plt.savefig(
                    f"C:\\Users\\Daniil\\Documents\\GitHub\\Dissertation\\data\\__{i}.png",
                    dpi=1200,
                )
                plt.show()

        elif len(coefs.shape) == 1:
            df = pd.DataFrame(coefs).T
            if col_names:
                df.columns = col_names
            else:
                df.columns = estimator[:-1].get_feature_names_out()
            df.index = ["Степень важности"]
            plt.figure().set_size_inches(16, 4)
            plt.title(
                f"Коэффициенты важности признаков для предсказания неисправностей {d[0]}"
            )
            sns.barplot(abs(df))
            plt.xticks(rotation=90)
            plt.xticks(fontsize=8)
            plt.savefig(
                f"C:\\Users\\Daniil\\Documents\\GitHub\\Dissertation\\data\\{i}.png",
                dpi=1200,
            )
            plt.show()

        else:
            print(len(coefs))

    def show_stat_results() -> pd.DataFrame:
        """Чтение файла статистики работы моделей"""

        data = pd.read_csv("stat_results.csv")
        columns = data.columns

        cols_to_drop = []

        if "Unnamed: 0" in columns:
            print("in")
            for c in columns:
                if "Unnamed" in c:
                    cols_to_drop.append(c)
            # print(cols_to_drop)
            return data.drop(cols_to_drop, axis=1)

        elif "Unnamed: 0" not in columns:
            return data

    def pretty_save_estimator_result(
        estimator, X_test: pd.DataFrame, y_test: pd.Series
    ):
        """Сохранение результатов модели в виде таблицы для отчетов"""

        if type(estimator[1]) == xgboost.sklearn.XGBClassifier:
            save_path = "/xgb_models_report"

        elif type(estimator[1]) == lightgbm.sklearn.LGBMClassifier:
            save_path = "/lgbm_models_report"

        elif type(estimator[1]) == catboost.core.CatBoostClassifier:
            save_path = "/catboost_models_report"

        else:
            raise ValueError("Передан не бустинг!")

        rep = classification_report(
            y_true=y_test, y_pred=estimator.predict(X_test), digits=6, output_dict=True
        )

        pd.DataFrame(rep).T.to_excel(
            f"{save_path}/report_from_{datetime.datetime.now()}.excel"
        )
