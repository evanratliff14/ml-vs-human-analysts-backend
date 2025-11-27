from abc import abstractmethod
from model import Model
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector
import copy
import logging
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nflreadpy as nfl
from sklearn.inspection import permutation_importance



class Seasonal(Model):
    def __init__(self, points_type, position, type = 'xgb'):
        logging.info("Initializing...")
        super().__init__(points_type)
        self.type = type
        self.position = position

        train_test_data = self.train_test_data
        eval = self.eval
        test_size = 0.2
        features = [line.strip() for line in open(f"data/{position.lower()}_features.txt", "r")]
        self.features = features

        

        train_test_data, eval= train_test_data.loc[train_test_data['position'] == position], eval.loc[eval['position'] == position]


        mask = train_test_data['season'] < nfl.get_current_season()-2
        train = train_test_data.loc[mask]
        test = train_test_data.loc[~mask]
        X_train, X_test = train[features], test[features]
        y_train, y_test = train[self.label], test[self.label]

        # X_train, X_test, y_train, y_test = train_test_split(train_test_data[features], train_test_data[self.label], 
        #     test_size=test_size, random_state = 42)
        
        logging.info(f"Train and test data has {(1-test_size)*train_test_data.shape[0]} train rows and {(test_size)*train_test_data.shape[0]} test rows")

        X_eval = eval[features]
        y_eval = eval[self.label]

        # create new dfs for rejoining
        train = pd.DataFrame()
        test = pd.DataFrame()
        eval = pd.DataFrame()
        
        if os.path.isfile(f'cache/{self.position.lower()}_test.parquet') and os.path.isfile(f'cache/{self.position.lower()}train.parquet') and os.path.isfile(f'cache/{self.position.lower()}_eval.parquet'):
            train = pd.read_parquet(f'cache/{self.position}_train.parquet').fillna(0, inplace=False, index=False)
            test = pd.read_parquet(f'cache/{self.position}_test.parquet').fillna(0, inplace=False, index=False)
            eval = pd.read_parquet(f'cache/{self.position}_eval.parquet').fillna(0, inplace=False, index=False)
#LEGACY 
            # train[self.label] = self.y_train
            # test[self.label]  = self.y_test
            # eval[self.label] = self.eval[self.label]
            # self.eval = eval
            # self.train = train
            # self.test = test
        else:

            logging.info("Imputing missing values...")
            imputer =IterativeImputer(max_iter=500, n_nearest_features=5, 
                initial_strategy='median', random_state=42, 
                add_indicator=False)
            imputer.set_output(transform = 'pandas')

            numeric = [feat for feat in self.features if feat not in self.categorical_identifiers and feat !=self.label]
            
            # Fit on train numeric only
            imputer.fit(X_train[numeric])

            # Transform both (use .loc to avoid accidental reindexing)
            X_train.loc[:, numeric] = imputer.transform(X_train[numeric])
            X_test.loc[:, numeric]  = imputer.transform(X_test[numeric])
            X_eval.loc[:, numeric] = imputer.transform(X_eval[numeric])

            # rejoin
            train[list(X_train.columns)] = X_train[list(X_train.columns)]
            test[list(X_test.columns)] = X_test[list(X_test.columns)]
            eval[list(X_eval.columns)] = X_eval[list(X_eval.columns)]
            train[self.label] = y_train
            test[self.label] = y_test
            eval[self.label] = y_eval
            


            # Save parquets without the pandas index column
            train.to_parquet(f'cache/{self.position.lower()}_train.parquet', index=False )
            test.to_parquet(f'cache/{self.position.lower()}_test.parquet', index=False)
            eval.to_parquet(f'cache/{self.position.lower()}_eval.parquet', index=False)
            
    
        self.train = train
        self.test = test
        self.eval = eval
        self.set_model()
        logging.info("Model is ready to train")

    def set_model(self):
        # n estimators is number of trees in the ensemble
        # use max leaf nodes instead of max depth??
        # validation_fraction is 0.0 because we are using our own cross validation methods
        # can use impurity decreate, max depth, or max leaf nodes. impurity decrease measures the MSE loss of a node
        # can use a combo of size and impurity based limits
        # n_iter_no_change, validation fraction focus on early stopping (validation fraction only used is n is integer)
        type = self.type
        if type == 'xgb':
            self.model = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=500, subsample = 0.95,
                criterion='friedman_mse', min_samples_split=250, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_depth=7, min_impurity_decrease=64.0, init=None, random_state=42, max_features=0.95, alpha=0.9, 
                verbose=0, max_leaf_nodes=128, warm_start=False, validation_fraction=0.001, n_iter_no_change=None,
                tol=0.1, ccp_alpha=0.0)

    # may pass anything that uses model interface, including sequential feature selector
    def train_model(self, model, features = None):
        logging.info("Training model")
        if features is None:
            features = self.features
        
        # remove categorical identifiers
        features = [f for f in features if f not in self.categorical_identifiers]

        model.fit(self.train[features], self.train[self.label])
        joblib.dump(model, f"{self.position}_{self.type}.joblib")
        self.model = model
        logging.info("Training complete")


    def test_model(self, features=None):
        logging.info("Testing model...")
        if features is None:
            features = self.features

        features = [f for f in features if f not in self.categorical_identifiers]

        # staged predict: returns each stage of the prediction of the test set, vs just the final
        self.train['predictions'] = self.model.predict(self.train[features])
        self.test['predictions'] = self.model.predict(self.test[features])
        self.eval['predictions'] = self.model.predict(self.eval[features])
        # self.eval['predictions'] = self.model.predict(self.eval[features])
                
        stage_errors = []
        for i, y_pred_stage in enumerate(self.model.staged_predict(self.test[features])):
            mse = mean_squared_error(self.test[self.label], y_pred_stage)
            stage_errors.append(mse)
            logging.info(f"Iteration {i+1}: MSE = {mse}")
        

    def set_features(self):
        logging.info("Setting features...")
        if not os.path.isfile(f'data/{self.position.lower()}_features.txt'):
            return Exception(f"{self.position}_features.txt does not exist in filepath!")
        else:
            features = [line.strip() for line in open(f"data/{self.position.lower()}_features.txt")]
            self.features = features
        logging.info("Features set.")

    def __str__(self):
        return super().__str__()
        
    def cross_validate(self):
        try:
            self.test.sort_values(
            by='predictions',
            ascending = False,
            inplace=False,
            kind = 'stable'
              # descending predictions, ascending season
            )     
            self.train.sort_values(
            by='predictions',
            ascending = False,
            inplace=False,
            kind = 'stable'
              # descending predictions, ascending season
            )     
            self.eval.sort_values(
            by='predictions',
            ascending = False,
            inplace=False,
            kind = 'stable'
              # descending predictions, ascending season
            )     
            self.test_mse = mean_squared_error(self.test['predictions'].iloc[0:100], self.test[self.label].iloc[0:100])
            self.test_mae = mean_absolute_error(self.test['predictions'].iloc[0:100],self.test[self.label].iloc[0:100])
            self.train_mse = mean_squared_error(self.train['predictions'].iloc[0:100],self.train[self.label].iloc[0:100])
            self.train_mae = mean_absolute_error(self.train['predictions'].iloc[0:100],self.train[self.label].iloc[0:100])
            self.eval_mse = mean_squared_error(self.eval['predictions'].iloc[0:100],self.eval[self.label].iloc[0:100])
            self.eval_mae = mean_absolute_error(self.eval['predictions'].iloc[0:100],self.eval[self.label].iloc[0:100])
        except Exception as e:
            print("Error in cross_validate" + str(e))

    def corr(self):
        fantasy_data = self.fantasy_data
        features = [feat for feat in self.features if feat not in self.categorical_identifiers]

        plt.figure(figsize=(20, 10))  # width, height in inches

        sns.heatmap(fantasy_data[features].corr(numeric_only=True).abs(), cmap='coolwarm')
        plt.savefig('corr_plot.png')
        plt.show()

    def __str__(self):
        model_string = "Features: \n"
        model_string = model_string + str(self.features) + "\n"
        # intentional side effect
        display = self.eval.copy()
        display['season'] = display['season'] +1
        display = display[['headshot_url', 'player_name', 'predictions', 'season', self.label]].sort_values(
            by='predictions',
            ascending = False,
            inplace=False,
            kind = 'stable'
              # descending predictions, ascending season
        )       
        display = display.sort_values(
            by='season',
            ascending = False,
            inplace=False,
            kind='stable'
              # descending predictions, ascending season
        )  
        display = display.drop_duplicates(subset=['player_name', 'season'], keep='first')
        logging.info(display)

        
        display.to_parquet(f'data/{self.position.lower()}_predictions_{self.type}.parquet')
        self.cross_validate()
        model_string += "Test MSE: " + str(self.test_mse) + "\n"
        model_string += "Test MAE: " + str(self.test_mae) + "\n"
        model_string += "Train MSE: " + str(self.train_mse) + "\n"
        model_string += "Test MAE: " + str(self.train_mae) + "\n"
        model_string += "Eval MSE: " + str(self.eval_mse) + "\n"
        model_string += "Eval MAE: " + str(self.eval_mae) + "\n"
        
        with open(f'data/{self.position}_error.txt', 'w') as file:
            file.write("Train RMSE: " + str(self.train_mse**(1/2)) + "\n")
            file.write("Train MAE: " + str(self.train_mae) + "\n")
            file.write("Eval RMSE: " + str(self.eval_mse**(1/2)) + "\n")
            file.write("Eval MAE: " + str(self.eval_mae) + "\n")

        features = [feat for feat in self.features if feat not in self.categorical_identifiers]

        r = permutation_importance(self.model, self.eval[features], self.eval[self.label],
                                n_repeats=1,
                                random_state=0,n_jobs=-1)
        # Pretty print only features that are “significant” by your chosen threshold
        print("Eval permutation importance")
        sorted_idx = r.importances_mean.argsort()[::-1]

        for i in sorted_idx:
            mean = r.importances_mean[i]
            std = r.importances_std[i]
            # heuristic: mean is reliably > 0 (2-sigma rule); adjust multiplier if you want
            if mean - 2 * std > 0:
                print(f"{features[i]:<30} {mean:.4f} +/- {std:.4f}")

        s = permutation_importance(self.model, self.test[features], self.test[self.label],
                                n_repeats=1,
                                random_state=0,n_jobs=-1)

        # Pretty print only features that are “significant” by your chosen threshold
        sorted_idx = r.importances_mean.argsort()[::-1]
        print("Test permutation importance")
        with open(f'data/{self.position}_perm_importance.txt', 'w') as file:
            for i in sorted_idx:
                mean = s.importances_mean[i]
                std = s.importances_std[i]
                # heuristic: mean is reliably > 0 (2-sigma rule); adjust multiplier if you want
                if mean - 2 * std > 0:
                    file.write(f"{features[i]:<30} {mean:.4f} +/- {std:.4f}")


        return model_string

        
