from pipelines.catboost_pipe import CatBoostPipe

MODELS_TO_BUILD=(
    CatBoostPipe,
)

def run_pipeline():
    """
    Run the pipeline for all models.
    """
    for model in MODELS_TO_BUILD:
        pipe = model()
        pipe.preprocess()
        pipe.train_models()