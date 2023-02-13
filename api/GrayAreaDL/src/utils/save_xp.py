import os
from pathlib import Path
from shutil import copyfile
import pandas as pd
import logging
import joblib

class SaveXp():
    """
    Allows you to save your experiences. The average is calculated between the metrics obtained during each split of the cross-validation
    """

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.score_cv = dict()
        self.probas_cv = dict()

    def save_experiment_metadata(self, yaml_config_path):
        os.makedirs(self.log_dir, exist_ok=True)
        yml_filename = self.log_dir / Path(yaml_config_path).name
        copyfile(yaml_config_path, yml_filename)
        # git
        try:
            with open(self.log_dir / Path("git_hash_commit.txt"), "w") as text_file:
                text_file.write(str(self.hash))
        except Exception:
            logging.error("can't save hash commit")

    def save_pipeline(self, pipeline_prepro,name=None):
        pipeline_path = self.log_dir / f'preprocessing{name}.joblib'
        joblib.dump(pipeline_prepro, pipeline_path)


if __name__ == "__main__":
    pipeline = joblib.load("./preprocessing.joblib")
