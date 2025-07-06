from mixins.preprocessor import Preprocessor
from mixins.trainer import Trainer
from catboost import CatBoostClassifier

class CatBoostPipe(Preprocessor,Trainer):
    MODEL_NAME = "catboost"
    MODEL = CatBoostClassifier
    
    def __init__(self):
        super().__init__()
        