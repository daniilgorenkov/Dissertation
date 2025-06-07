from mixins.preprocessor import Preprocessor
from mixins.trainer import Trainer

class CatBoostPipe(Preprocessor,Trainer):
    MODEL_NAME = "catboost"
    def __init__(self):
        super().__init__()
        