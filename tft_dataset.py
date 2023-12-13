import warnings
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, NaNLabelEncoder
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

max_prediction_length = 10
max_encoder_length = 168


def make_timeseries_dataset(dataset):
    """
    Make the dataset for hourly training
    :param dataset: dataset that was loaded via preprocessing.TFTDataset from a csv file
    :return: dataset for hourly training
    """
    training_cutoff = dataset["time_idx"].max() - max_prediction_length
    if 'Volume' in dataset.columns:
        dataset['Volume'] = np.log2(dataset['Volume'])
        dataset.replace([np.inf, -np.inf], 0, inplace=True)
    tft_data = TimeSeriesDataSet(
        dataset[lambda x: x.time_idx <= training_cutoff],
        allow_missing_timesteps=True,
        time_idx='time_idx',
        target="Close",
        group_ids=[
            "StatSymbol",
        ],
        min_encoder_length=max_encoder_length//2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["StatSymbol"],
        time_varying_known_categoricals=["DayOfWeek", "DayOfMonth", "HourOfDay"],
        time_varying_unknown_reals=[
            "Volume",
            "Close",
            "RSI",
        ],
        categorical_encoders={'StatSymbol': NaNLabelEncoder(add_nan=True),
                              'DayOfWeek': NaNLabelEncoder(add_nan=True),
                              'DayOfMonth': NaNLabelEncoder(add_nan=True),
                              'HourOfDay': NaNLabelEncoder(add_nan=True),
                              'Volume': NaNLabelEncoder(add_nan=True),
                              'Close': NaNLabelEncoder(add_nan=True),
                              'RSI': NaNLabelEncoder(add_nan=True),
                              },

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return tft_data


def make_daily_timeseries_dataset(dataset):
    """
    Make the dataset for daily training
    :param dataset: dataset that was loaded via preprocessing.DailyTFTDataset from a csv file
    :return: dataset daily training
    """
    training_cutoff = dataset["time_idx"].max() - max_prediction_length
    if 'Volume' in dataset.columns:
        dataset['Volume'] = np.log2(dataset['Volume'])
        dataset.replace([np.inf, -np.inf], 0, inplace=True)
    tft_data = TimeSeriesDataSet(
        dataset[lambda x: x.time_idx <= training_cutoff],
        allow_missing_timesteps=True,
        time_idx='time_idx',
        target="Close",
        group_ids=[
            "StatSymbol",
        ],
        min_encoder_length=max_encoder_length//2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["StatSymbol"],
        time_varying_known_categoricals=["DayOfWeek", "DayOfMonth", "WeekOfYear"],
        time_varying_unknown_reals=[
            "Volume",
            "Close",
            "RSI",
        ],
        categorical_encoders={'StatSymbol': NaNLabelEncoder(add_nan=True),
                              'DayOfWeek': NaNLabelEncoder(add_nan=True),
                              'DayOfMonth': NaNLabelEncoder(add_nan=True),
                              'WeekOfYear': NaNLabelEncoder(add_nan=True),
                              'Volume': NaNLabelEncoder(add_nan=True),
                              'Close': NaNLabelEncoder(add_nan=True),
                              'RSI': NaNLabelEncoder(add_nan=True),
                              },
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return tft_data

