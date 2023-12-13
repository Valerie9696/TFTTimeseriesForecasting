import os
import lightning.pytorch as pl
import torch.cuda
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet, NaNLabelEncoder
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


def make_trainer(folder_name, number_train_batches=500):
    """
    Make the trainer and store its output at the given path (folder_name)
    default uses 500 train batches for hourly training. For daily training, give 100.

    :param folder_name: path to the folder where the output will be stored
    :param number_train_batches: number of training batches (500 for hourly, 100 for daily)
    :return: Trainer
    """
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
    model_checkpoint_callback = ModelCheckpoint(dirpath=folder_name, filename='tft_checkpoint', save_top_k=1)
    accelerator = "cpu"
    if torch.cuda.is_available():
        accelerator = "gpu"
    trainer = pl.Trainer(
        default_root_dir=folder_name,
        max_epochs=100,
        accelerator=accelerator,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=number_train_batches,  # 500 for hourly training, 100 for daily
        callbacks=[lr_logger, early_stop_callback, model_checkpoint_callback],
        logger=logger,
    )
    return trainer


def make_model(dataset, period = 'hourly'):
    """
    make the temporal fusion transformer model and return it
    :param dataset: tft dataset
    :return: temporal fusion transformer model
    """
    params = None
    if period == 'hourly':
        params = {'gradient_clip_val': 0.15, 'hidden_size': 36, 'dropout': 0.1,
                    'hidden_continuous_size': 10, 'attention_head_size': 1, 'learning_rate': 0.064}
    elif period == 'daily':
        params = {'gradient_clip_val': 0.02, 'hidden_size': 81, 'dropout': 0.15,
                     'hidden_continuous_size': 16, 'attention_head_size': 4, 'learning_rate': 0.02}
    if params is not None:
        tft = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=params['learning_rate'],
            hidden_size=params['hidden_size'],
            attention_head_size=params['attention_head_size'],
            dropout=params['dropout'],
            hidden_continuous_size=params['hidden_continuous_size'],
            loss=QuantileLoss(),
            log_interval=0,
            optimizer="Ranger",
            reduce_on_plateau_patience=4,
        )
    else:
        print('No valid period was given. TFT will be initialized with placeholder values.')
        tft = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=0.1,
            hidden_size=64,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=28,
            loss=QuantileLoss(),
            log_interval=0,
            optimizer="Ranger",
            reduce_on_plateau_patience=4,
            )
    return tft
