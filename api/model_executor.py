from fantasy_df import FantasyDataFrame
from seasonal import Seasonal
import logging
import pyarrow
import os

class ModelExecutor:
    def __init__(self):
        if not os.path.isfile('data/data.parquet'):
            fdf = FantasyDataFrame()
            self.fdf = fdf
            logging.info("Creating parquet...")
            fdf.players_stats.to_parquet('data/data.parquet', index=False)
        
        self.rb_seasonal = Seasonal(points_type='ppr', position = 'RB', type = 'xgb')
        self.qb_seasonal = Seasonal(points_type='ppr', position = 'QB', type = 'xgb')
        self.te_seasonal = Seasonal(points_type='ppr', position = 'TE', type = 'xgb')
        self.wr_seasonal = Seasonal(points_type='ppr', position = 'WR', type = 'xgb')


    def run(self):
        rb_seasonal = self.rb_seasonal
        rb_seasonal.corr()

        # only outputting standard/game right now
        rb_seasonal.set_features()
        rb_seasonal.train_model(rb_seasonal.model)
        rb_seasonal.test_model()
        rb_seasonal.cross_validate()

        
        qb_seasonal = self.qb_seasonal
        qb_seasonal.corr()

        # only outputting standard/game right now
        # rb_seasonal.set_features()
        qb_seasonal.train_model(qb_seasonal.model)
        qb_seasonal.test_model()
        qb_seasonal.cross_validate()

        te_seasonal = self.te_seasonal
        te_seasonal.corr()

        # only outputting standard/game right now
        te_seasonal.set_features()
        te_seasonal.train_model(te_seasonal.model)
        te_seasonal.test_model()
        te_seasonal.cross_validate()

        wr_seasonal = self.wr_seasonal
        wr_seasonal.corr()

        # only outputting standard/game right now
        # wr_seasonal.set_features()
        wr_seasonal.train_model(wr_seasonal.model)
        wr_seasonal.test_model()
        wr_seasonal.cross_validate()



        print(rb_seasonal)
        print(te_seasonal)
        print(wr_seasonal)
        print(qb_seasonal)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ModelExecutor().run()