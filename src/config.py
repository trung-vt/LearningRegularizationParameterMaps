from dataclasses import dataclass

@dataclass
class Paths:
    data_path: str
    model_path: str
    log_path: str
    output_path: str

@dataclass
class Files:
    train_file: str
    test_file: str
    model_file: str
    log_file: str
    output_file: str

@dataclass
class Params:
    """
    Hyperparameters
    """
    epoch_count: int
    learning_rate: float
    batch_size: int
    T_train: int # Number of iterations for PDHG during training
    up_bound: float # Bound the regularisation parameters somehow?