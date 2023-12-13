import os
import warnings
import lightning.pytorch as pl
import pandas as pd
import pickle
import training_handler
from pt_f.pytorch_forecasting.metrics import MAE
from pytorch_forecasting import Baseline, TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import preprocessing
import tft_dataset
from lightning.pytorch.tuner import Tuner
period = "daily"

# use this tuner for hyperparameter tuning

warnings.filterwarnings("ignore")  # avoid printing out absolute paths

max_prediction_length = 10
max_encoder_length = 168
# make dataset fitting the kind of data (daily/ hourly)
if period == "hourly":
    df = preprocessing.TFTDataset(path=os.path.join('ohlc', 'ohlc.csv'))
    train_set = pd.concat([df.train_set, df.valid_set])
    training = tft_dataset.make_timeseries_dataset(train_set)
    validation = TimeSeriesDataSet.from_dataset(training, train_set, predict=True, stop_randomization=True)
else:
    df = preprocessing.DailyTFTDataset(path=os.path.join('ohlc', 'daily_ohlc.csv'))
    train_set = pd.concat([df.train_set, df.valid_set])
    training = tft_dataset.make_daily_timeseries_dataset(dataset=train_set)
    validation = TimeSeriesDataSet.from_dataset(training, train_set, predict=True, stop_randomization=True)

# make the dataloaders
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0)

baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
MAE()(baseline_predictions.output, baseline_predictions.y)

if not os.path.isdir('Tuning'):
    os.mkdir("Tuning")
pl.seed_everything(42)

if period == 'hourly':
    trainer = training_handler.make_trainer(folder_name="Tuning")
else:
    trainer = training_handler.make_trainer(folder_name="Tuning", number_train_batches=100)
tft = training_handler.make_model(training)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find a suitable learning rate
res = Tuner(trainer).lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# create study for hyperparameter tuning
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path=os.path.join('Tuning', 'tuning_model'),
    n_trials=200,
    max_epochs=25,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 96),
    hidden_continuous_size_range=(8, 96),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.01, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,
)

# save tuning results
with open(os.path.join("Tuning", period+"_test_study.pkl"), "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)

# result of the latest run for hourly:
#{'gradient_clip_val': 0.14510904446106018, 'hidden_size': 36, 'dropout': 0.10840386932063481,
# 'hidden_continuous_size': 10, 'attention_head_size': 1, 'learning_rate': 0.06411188635050383}

# daily:
# {'gradient_clip_val': 0.019521186757910997, 'hidden_size': 81, 'dropout': 0.14254373657604905,
# 'hidden_continuous_size': 16, 'attention_head_size': 4, 'learning_rate': 0.01887425748950314}