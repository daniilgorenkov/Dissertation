from pipelines.catboost_pipe import CatBoostPipe


def run_pipeline():
    """
    Run the pipeline for all models.
    """
    model = CatBoostPipe()
    model.preprocess()
    model.train_models()

if __name__=="__main__":
    run_pipeline()