import config
import pandas as pd
import numpy as np
from tqdm import tqdm
import io
import contextlib
import gc
from mixins.utils import set_logger
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score,accuracy_score, precision_score, recall_score,multilabel_confusion_matrix
import argparse
from mixins.file_operator import FileOperator
from sklearn.model_selection import train_test_split

logger = set_logger(config.Paths._LOGS)

# Fix seed
seed = config.Common.SEED
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--update_params", action="store_true")  # environment, For prod only need this.
args, _ = parser.parse_known_args()

class ForcesDataset:
    def __init__(self, df:pd.DataFrame):
        self.df = df
    
    def create_dataset(self,target:str) -> dict[pd.DataFrame, pd.Series]:
        """
        Create a dataset from the DataFrame.
        """
        ds = {}
        X = self.df.drop(columns=[target])
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.Common.SEED)
        X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=config.Common.SEED)
        ds['X_train'] = X_train
        ds['y_train'] = y_train
        ds['X_dev'] = X_dev
        ds['y_dev'] = y_dev
        ds['X_test'] = X_test
        ds['y_test'] = y_test
        ds["cat_cols"] = [col for col in X.columns if X[col].dtype == "int32" and col not in config.Preprocessor.TARGETS]
        ds["cont_cols"] = [col for col in X.columns if X[col].dtype == "float32"]
        return ds
    

class Trainer(FileOperator):
    def __init__(self):
        super().__init__()

    
    def model_fit(self,
                  model:CatBoostClassifier,
                  train_pool:Pool,
                  dev_pool:Pool,
                  params:dict):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            model = model(**params)
            model.fit(train_pool, eval_set=dev_pool, use_best_model=True)
        
        logs = buffer.getvalue().strip().split("\n")
        for i, line in enumerate(logs):
            if i % config.Trainer.PRINT_ITERATION == 0 or i == len(logs) - 1:  # Cada 100 líneas + última línea
                logger.debug(f"[training]: {line}")
        return model      

    def log_model_results(self,
                      ds: dict,
                      model: CatBoostClassifier,
                      model_task: str = "fault_target"):

        models_scores = {}
        logger.debug(f"{model_task.upper()} model results:")
        logger.debug(f"{'-'*128}")
        logger.debug(f"{'split':>12} | {'log_loss':>9} | {'std':>9} | {'min':>9} | {'max':>9} | {'auc_roc':>9} | {'accuracy':>9} | {'precision':>9} | {'recall':>9} | {'rows':>9} |")  # fmt: skip
        logger.debug(f"{'-'*128}")

        X_list = ['X_train', 'X_dev', 'X_test']
        y_list = ['y_train', 'y_dev', 'y_test']
        for x_, y_ in zip(X_list, y_list):
            x, y = ds[x_], ds[y_]
            preds = model.predict_proba(x)

            split_name = x_.split("_")[-1]

            if preds.shape[1] == 2:
                # Binary classification
                if y.ndim == 2:
                    y = np.argmax(y, axis=1)
                y_pred = (preds[:, 1] > 0.5).astype(int)
                auc = roc_auc_score(y, preds[:, 1])
            else:
                # Multiclass classification
                if y.ndim == 2:
                    y = np.argmax(y, axis=1)
                y_pred = np.argmax(preds, axis=1)
                auc = roc_auc_score(y, preds, multi_class='ovr')

            models_scores[split_name] = {
                "std": np.std(preds),
                "log_loss": log_loss(y, preds),
                "min": preds.min(),
                "max": preds.max(),
                "auc_roc": auc,
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, average='macro'),
                "recall": recall_score(y, y_pred, average='macro'),
                "rows": len(y)
            }

            logger.debug(
                f"{split_name:>12} | "
                f"{models_scores[split_name]['log_loss']:>9.4f} | {models_scores[split_name]['std']:>9.4f} | "
                f"{preds.min():>9.4f} | {preds.max():>9.4f} | {models_scores[split_name]['auc_roc']:>9.4f} | "
                f"{models_scores[split_name]['accuracy']:>9.4f} | {models_scores[split_name]['precision']:>9.4f} | {models_scores[split_name]['recall']:>9.4f} |"
                f"{len(y):>9} |"
            )

        logger.debug(f"{'-'*128}")

    def train_models(self):
        
        logger.debug(
            f"[ {self.MODEL_NAME.upper()} ] "
            f"DEVICE: {config.Trainer.DEVICE}")

        # Load datasets
        fault_dataset = ForcesDataset(self.load("preprocessed_data_fault_target")).create_dataset("fault_target")
        profile_dataset = ForcesDataset(self.load("preprocessed_data_profile_target")).create_dataset("profile_target")
        logger.debug(f"Fault dataset shape: {fault_dataset['X_train'].shape}, Profile dataset shape: {profile_dataset['X_train'].shape}")
        
        # create pools for fault target
        fault_train_pool = Pool(fault_dataset["X_train"], fault_dataset["y_train"], cat_features=fault_dataset["cat_cols"])
        fault_dev_pool = Pool(fault_dataset["X_dev"], fault_dataset["y_dev"], cat_features=fault_dataset["cat_cols"])
        
        # create pools for profile target
        profile_train_pool = Pool(profile_dataset["X_train"], profile_dataset["y_train"], cat_features=profile_dataset["cat_cols"])
        profile_dev_pool = Pool(profile_dataset["X_dev"], profile_dataset["y_dev"], cat_features=profile_dataset["cat_cols"])

        def objective_fault(trial):
            params = suggest_params(trial)
            params['loss_function'] = 'Logloss'
            params['eval_metric'] = 'Logloss'
            model = CatBoostClassifier(**params)
            logger.debug(f"Training fault model with params: {params}")
            model.fit(fault_train_pool, eval_set=fault_dev_pool, use_best_model=True)
            preds = model.predict_proba(fault_dataset["X_dev"])
            return log_loss(fault_dataset["y_dev"], preds)


        def objective_profile(trial):
            params = suggest_params(trial)
            params['loss_function'] = 'MultiClass'
            params['eval_metric'] = 'MultiClass'
            model = CatBoostClassifier(**params)
            logger.debug(f"Training profile model with params: {params}")
            model.fit(profile_train_pool, eval_set=profile_dev_pool, use_best_model=True)
            preds = model.predict(profile_dataset["X_dev"])
            return 1.0 - accuracy_score(profile_dataset["y_dev"], preds)
        
        def suggest_params(trial):
            
            return {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 1, 16),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'task_type': 'GPU' if config.Trainer.DEVICE == "cuda" else "CPU",
                'verbose': 1,
                'early_stopping_rounds': 15,
                'random_seed': config.Common.SEED
            }
        
         
        if args.update_params:
            
            logger.debug(f"Start updating models params")
            pbar = tqdm(total=2, desc="Searching for best params", unit="model")
            # searching for best params for model
            study_fault = optuna.create_study(direction="minimize")
            study_fault.optimize(objective_fault, n_trials=config.Trainer.N_TRIALS)
            pbar.update(1)
            logger.debug(f"Best searched params:\n{study_fault.best_params}")

            study_profile = optuna.create_study(direction="minimize")
            study_profile.optimize(objective_profile, n_trials=config.Trainer.N_TRIALS)
            pbar.update(1)
            pbar.close()
            logger.debug(f"Best searched params:\n{study_profile.best_params}")

            # train fault model with best params
            pbar = tqdm(total=2, desc="Training best models", unit="model")
            model_fault = CatBoostClassifier(**study_fault.best_params)
            model_fault.fit(fault_train_pool, eval_set=fault_dev_pool, use_best_model=True)
            self.save(model_fault,"fault_model")
            pbar.update(1)
            # model_fault = self.load("fault_model")
            # print logs with stats
            self.log_model_results(fault_dataset,model_fault)

            # train profile model with best params
            model_profile = CatBoostClassifier(**study_fault.best_params)
            model_profile.fit(profile_train_pool, eval_set=profile_dev_pool, use_best_model=True)
            self.save(model_profile,"profile_model")
            pbar.update(1)
            self.log_model_results(profile_dataset,model_profile,"profile_target")
            pbar.close()

        else:
            logger.debug(f"Start training models with default params")
            # fault model training
            pbar = tqdm(total=2, desc="Training best models", unit="model")
            model_fault = self.model_fit(CatBoostClassifier,fault_train_pool,fault_dev_pool,config.Trainer.FAULT_BOOST_PARAMS)
            self.save(model_fault,"fault_model")
            # model_fault = self.load("fault_model")
            pbar.update(1)
            # print logs with stats
            self.log_model_results(fault_dataset,model_fault)

            # profile model training
            model_profile = self.model_fit(CatBoostClassifier,profile_train_pool,profile_dev_pool,config.Trainer.PROFILE_BOOST_PARAMS)   
            self.save(model_profile,"profile_model")
            # model_fault = self.load("fault_model")
            pbar.update(1)
            pbar.close()
            # print logs with stats
            self.log_model_results(profile_dataset,model_profile,"profile_target")

        gc.collect()
