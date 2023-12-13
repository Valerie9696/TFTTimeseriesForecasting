import os
import warnings
import lightning.pytorch as pl
import pandas as pd
import torch
torch.set_float32_matmul_precision('medium')

from pytorch_forecasting import TimeSeriesDataSet
import preprocessing
import tft_dataset
import training_handler
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# Training script for daily ohlc data
try:
    df = preprocessing.DailyTFTDataset(path=os.path.join('ohlc', 'daily_ohlc.csv'))
except:
    print('File not found.')
train_set = pd.concat([df.train_set, df.valid_set])
training = tft_dataset.make_daily_timeseries_dataset(dataset=train_set)
validation = TimeSeriesDataSet.from_dataset(training, train_set, predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0)
pl.seed_everything(42)
if not os.path.isdir('DailyCheckpoints'):
    os.mkdir('DailyCheckpoints')
trainer = training_handler.make_trainer(folder_name='DailyCheckpoints', number_train_batches=100)

tft = training_handler.make_model(dataset=training, period='daily')
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
