from abc import abstractmethod
from re import S
import pandas as pd
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import copy
import nflreadpy as nfl
import pyarrow
import logging

class Model:
    def __init__(self, points_type, **kwargs):
        logging.info("Reading data from parquet")
        self.label = f"future_{points_type}/game"
        self.categorical_identifiers = ('season', 'team', 'position', 'player_name', 'headshot_url')

        fantasy_data = pd.read_parquet('data/data.parquet')
        self.fantasy_data = fantasy_data

        #clean data at this step - we don't want to call fantasy_df too many times to clean data
        fantasy_data.dropna(subset=[self.label], axis=0, inplace=True)
        fantasy_data = fantasy_data.loc[(fantasy_data[self.label] >= 0) | (fantasy_data['season']>=nfl.get_current_season())]
        threshold = 0.3
        fantasy_data = fantasy_data.dropna(axis=1, thresh=len(fantasy_data)*threshold)

        #refactor for categorical features
        features = [feat for feat in list(fantasy_data.columns) if (pd.api.types.is_numeric_dtype(fantasy_data[feat]) or feat in self.categorical_identifiers)]
        features = [feat for feat in features if 'future' not in feat.lower()]

        logging.info(f"Total numeric columns and position {features}")
        current_data = fantasy_data.loc[fantasy_data['season'] == nfl.get_current_season()]

        current_data.to_csv('current_data.csv')

        eval_data = fantasy_data.loc[fantasy_data['season'] == nfl.get_current_season()-1 ]
        train_test_data = fantasy_data.loc[fantasy_data['season'] < nfl.get_current_season()-1 ]


        # should have categorical idenifiers, sparse vars, and lack future vars
        self.features = features
        self.eval = eval_data
        self.train_test_data = train_test_data
        
        self.points_type = points_type
        self.model = None

    @abstractmethod
    def train_model(self, model, features=None):
        pass

    @abstractmethod
    def test_model(self, features=None):
        pass
    
    @abstractmethod
    def cross_validate(self):
       pass
    
    @abstractmethod
    def __str__(self):
       pass

    @abstractmethod
    def set_features(self):
        pass