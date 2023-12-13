import warnings
import pandas as pd
import torch
import glob
import os
from pytorch_forecasting import TemporalFusionTransformer

import preprocessing
import tft_dataset
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
TORCH_USE_CUDA_DSA=1
# set the period in order to make predictions for daily or hourly data
period = 'hourly'

# use this script in order to make predictions

accelerator = "cpu"
if torch.cuda.is_available():
    accelerator = "gpu"

if not os.path.exists('Predictions'):
    os.mkdir('Predictions')

if period == 'daily':
    daily_files = glob.glob(os.path.join('DailyCheckpoints','*.ckpt'))
    latest_daily_ckpt = max(daily_files, key=os.path.getctime)
    df = preprocessing.DailyTFTDataset(path=os.path.join('ohlc', 'daily_ohlc.csv'))
    test_set = df.test_set.dropna()
    batch_size = 128
    row_count = len(test_set.index)
    # test set length needs to be dividable by batch size - this is hard coded and needs to be
    # adjusted if another dataset is used.
    drop_list = [x for x in range(row_count - 13, row_count)]
    test_set = test_set.drop(drop_list)
    test = tft_dataset.make_daily_timeseries_dataset(dataset=test_set)
    test_dataloader = test.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(latest_daily_ckpt)
    best_tft.accelerator = accelerator
    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    predictions = best_tft.predict(test_dataloader, mode='prediction', return_index=True, return_x=True, return_y=True)
    df = pd.DataFrame(predictions.output.data.cpu())
    # save data to csv
    path = os.path.join('Predictions', 'daiy_predictions.csv')
    df.to_csv(path)
else:
    # in this case, predict on hourly data
    files = glob.glob(os.path.join('Checkpoints','*.ckpt'))
    latest_ckpt = max(files, key=os.path.getctime)
    df = preprocessing.TFTDataset(path=os.path.join('ohlc', 'ohlc.csv'))
    test_set = df.test_set.dropna()
    row_count = len(test_set.index)
    # test set length needs to be dividable by batch size - this is hard coded and needs to be
    # adjusted if another dataset is used.
    drop_list = [x for x in range(row_count-141, row_count)]
    test_set = test_set.drop(drop_list)
    test = tft_dataset.make_timeseries_dataset(dataset=test_set)
    # create dataloaders for model
    batch_size = 128
    test_dataloader = test.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    best_tft = TemporalFusionTransformer.load_from_checkpoint(latest_ckpt)
    predictions = best_tft.predict(test_dataloader, mode='prediction', return_index=True, return_x=True, return_y=True)
    df = pd.DataFrame(predictions.output.data.cpu())
    # save data to csv
    path = os.path.join('Predictions', 'predictions.csv')
    df.to_csv(path)


# comment in if you want to see some plots afterwards
#for idx in range(10):  # plot 10 examples
 #   fig = best_tft.plot_prediction(predictions.x, predictions.output, idx=idx, add_loss_to_title=True)
  #  fig.show()
