
from enum import Enum

class RUN_MODE(Enum):
    TRAIN = "train"
    TRAIN_CONTINUOUSLY = "train_continuously"
    EVALUATE_ONLY = "evaluate_only"

class CommonVars:
    def __init__(self) -> None:
        self.stats_dir = "stats"
        self.data_dir = "data"
        self.models_dir = ""
        self.model_name = ""
        
common_vars = CommonVars()