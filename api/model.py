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
    def __init__(self, points_type, cur_season, **kwargs):
        self.cur_season = cur_season
        logging.info("Reading data from parquet")
        self.label = f"future_{points_type}/game"
        self.categorical_identifiers = ('season', 'team', 'position', 'player_name', 'headshot_url')

        fantasy_data = pd.read_parquet('data/data.parquet')
        self.fantasy_data = fantasy_data

        # Keep inference-season rows even when future labels are missing (no next season yet).
        # Only require non-null, non-negative labels for historical (train/test) seasons.
        has_label = fantasy_data[self.label].notna() & (fantasy_data[self.label] >= 0)
        is_inference_season = fantasy_data['season'] >= cur_season
        fantasy_data = fantasy_data.loc[has_label | is_inference_season].copy()

        threshold = 0.3
        fantasy_data = fantasy_data.dropna(axis=1, thresh=len(fantasy_data)*threshold)

        #refactor for categorical features
        features = [feat for feat in list(fantasy_data.columns) if (pd.api.types.is_numeric_dtype(fantasy_data[feat]) or feat in self.categorical_identifiers)]
        features = [feat for feat in features if 'future' not in feat.lower()]

        logging.info(f"Total numeric columns and position {features}")
        # Prefer full seasonal rows over sparse weekly overlays for the same player-season.
        fantasy_data = fantasy_data.sort_values(
            by=[c for c in ['games', 'targets', 'carries', 'attempts'] if c in fantasy_data.columns],
            ascending=False,
            kind='stable',
        )
        fantasy_data = fantasy_data.drop_duplicates(
            subset=[c for c in ['player_id', 'position', 'season'] if c in fantasy_data.columns],
            keep='first',
        )

        current_data = fantasy_data.loc[fantasy_data['season'] == cur_season]

        current_data.to_csv('current_data.csv')

        eval_data = current_data
        train_test_data = fantasy_data.loc[fantasy_data['season'] < cur_season ]


        # should have categorical idenifiers, sparse vars, and lack future vars
        self.features = features
        self.eval = eval_data
        print(self.eval.size)
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