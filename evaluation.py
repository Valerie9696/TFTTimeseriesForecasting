import warnings
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import preprocessing
import tft_dataset
import training_handler

# script that loads the latest training checkpoint, predicts and plots 10 examples + the interpretation
# output which is a feature of the pytorch lightning implementation
# was used for overview right after moving from the nvidia implementation to the pytorch lightning one
# (only for hourly data), does not contribute much to the project

# find the latest training checkpoint
files = glob.glob(os.path.join('Checkpoints', '*.ckpt'))
latest_ckpt = max(files, key=os.path.getctime)
print(latest_ckpt)
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

df = preprocessing.TFTDataset(path=os.path.join('ohlc', 'ohlc.csv'))
train_set = pd.concat([df.train_set, df.valid_set])
training = tft_dataset.make_timeseries_dataset(dataset=train_set)
validation = TimeSeriesDataSet.from_dataset(training, train_set, predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0)
trainer = training_handler.make_trainer()

best_tft = TemporalFusionTransformer.load_from_checkpoint(latest_ckpt)

raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
# plot 10 examples
for idx in range(10):
    fig = best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
    fig.show()

predictions = best_tft.predict(val_dataloader, return_x=True)
#predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(predictions.x, predictions.output)
#best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
#plt.show()
interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
best_tft.plot_interpretation(interpretation)
plt.show()