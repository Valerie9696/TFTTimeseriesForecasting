import pandas as pd
import numpy as np
import utils
import base
import sklearn
import pickle as pkl

DataTypes = base.DataTypes
InputTypes = base.InputTypes

# taken from this repo https://github.com/greatwhiz/tft_tf2 (repo that was the base of the previous project)

class DailyTFTDataset:
    get_column_definition = [
        ('Symbol', DataTypes.CATEGORICAL, InputTypes.ID),
        ('Date', DataTypes.DATE, InputTypes.TIME),
        ('Open', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('High', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Low', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('HighLowDifference', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('OpenCloseDifference', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Volume', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ATR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('RSI', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Close', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('DaysFromStart', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('DayOfWeek', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('DayOfMonth', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('WeekOfYear', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('StatSymbol', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
    ]

    def __init__(self, path):
        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self.train_set, self.valid_set, self.test_set = self.split_dataset(path)

    def split_dataset(self, path):
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['WeekOfYear'] = df['WeekOfYear'].astype(str)
        df['DayOfWeek'] = df['DayOfWeek'].astype(str)
        df['DayOfMonth'] = df['DayOfMonth'].astype(str)
        if len(df) > 1:
            min_date = df['Date'].agg(['min'])[0]
            max_date = df['Date'].agg(['max'])[0]
            valid_boundary = min_date + (max_date - min_date) / 2
            test_boundary = valid_boundary + ((max_date - min_date) / 4)

            index = df['Date']
            train = df.loc[index < valid_boundary]
            valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
            test = df.loc[(index >= test_boundary)]  # & (df.index <= '2019-06-28')]
            print('Formatting train-valid-test splits.')
            self.set_scalers(df)

            return train, valid, test  # (self.transform_inputs(data) for data in [train, valid, test]) #train, valid, test
        else:
            return df  # self.transform_inputs(df) #df

    def set_scalers(self, df):
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        data = df[real_inputs].values
        scaler = sklearn.preprocessing.StandardScaler().fit(data)
        with open('scaler.pickle', 'wb') as f:
            pkl.dump(scaler, f)
        self._real_scalers = scaler
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values)  # used for predictions

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = srs  # .values # sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col]  # .transform(string_df)
        return output

    def format_predictions(self, predictions):
        output = predictions.copy()

        column_names = predictions.columns
        # output_items = output.items()

        for col in column_names:
            # for item in output_items:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(np.array(predictions[col]).reshape(-1, 1))

        return output

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 252 + 5,
            'num_encoder_steps': 252,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5,
        }

        return fixed_params


class TFTDataset:

    get_column_definition = [
      ('Symbol', DataTypes.CATEGORICAL, InputTypes.ID),
      ('Datetime', DataTypes.DATE, InputTypes.TIME),
      ('Open', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('High', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('Low', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('HighLowDifference', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('OpenCloseDifference', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('Volume', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ATR', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('RSI', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('Close', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('HourOfDay', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('HoursFromStart', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('DayOfWeek', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('DayOfMonth', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('WeekOfYear', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('StatSymbol', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
    ]


    def __init__(self, path):
        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self.train_set, self.valid_set, self.test_set = self.split_dataset(path)

    def split_dataset(self, path):
        df = pd.read_csv(path)
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        df['HourOfDay'] = df['HourOfDay'].astype(str)
        df['DayOfWeek'] = df['DayOfWeek'].astype(str)
        df['DayOfMonth'] = df['DayOfMonth'].astype(str)
        if len(df) > 1:
            min_date = df['Datetime'].agg(['min'])[0]
            max_date = df['Datetime'].agg(['max'])[0]
            valid_boundary = min_date + (max_date - min_date) / 2
            test_boundary = valid_boundary + ((max_date - min_date) / 4)

            index = df['Datetime']
            train = df.loc[index < valid_boundary]
            valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
            test = df.loc[(index >= test_boundary)]  # & (df.index <= '2019-06-28')]
            print('Formatting train-valid-test splits.')
            self.set_scalers(df)

            return train, valid, test#(self.transform_inputs(data) for data in [train, valid, test]) #train, valid, test
        else:
            return df#self.transform_inputs(df) #df

    def set_scalers(self, df):
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())


        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        data = df[real_inputs].values
        scaler = sklearn.preprocessing.StandardScaler().fit(data)
        with open('scaler.pickle', 'wb') as f:
            pkl.dump(scaler, f)
        self._real_scalers = scaler
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values)  # used for predictions

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
          # Set all to str so that we don't have mixed integer/string columns
          srs = df[col].apply(str)
          categorical_scalers[col] = srs#.values # sklearn.preprocessing.LabelEncoder().fit(srs.values)
          num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
          raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
          string_df = df[col].apply(str)
          output[col] = self._cat_scalers[col]#.transform(string_df)
        return output

    def format_predictions(self, predictions):
        output = predictions.copy()

        column_names = predictions.columns
        #output_items = output.items()

        for col in column_names:
        #for item in output_items:
          if col not in {'forecast_time', 'identifier'}:
            output[col] = self._target_scaler.inverse_transform(np.array(predictions[col]).reshape(-1, 1))

        return output

      # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 252 + 5,
            'num_encoder_steps': 252,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5,
        }

        return fixed_params
