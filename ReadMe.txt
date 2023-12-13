This is the final result of the timeseries forecasting isy project.
This repo contains only the final version for timeseries forecasting with a temporal fusion transformer
implementation provided by pytorch lightning. The previous version was based on this repo https://github.com/greatwhiz/tft_tf2
but since it never produced acceptable results even after some filtering and variable combination
experiments it is omitted here.

First install the requirements from the requirements.txt
In the last part of the project prediction was extended from hourly to daily data.
Use daily_trainer.py to train a model for daily ohlc data.
Use hourly_trainer.py to train a model for hourly ohlc data.
Use tuning.py to run hyperparameter tuning - specify on line 13 whether you
want the predictions for daily or hourly data
Use prediction.py in order to generate a csv file with predictions - specify on line 13 whether you
want the predictions for daily or hourly data; Uncomment the last lines in order to see some plots
at the end
Use trader.py in order to simulate trading based on the previously saved predictions.

