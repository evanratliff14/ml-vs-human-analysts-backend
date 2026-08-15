from fantasy_df import FantasyDataFrame
from seasonal import Seasonal
import logging
import pyarrow
import os
import argparse
import pandas as pd

import nflreadpy as nfl

class ModelExecutor:
    def __init__(self, cur_season):
        self.cur_season = cur_season
        parquet_path = 'data/data.parquet'
        needs_rebuild = not os.path.isfile(parquet_path)
        if not needs_rebuild:
            existing = pd.read_parquet(
                parquet_path, columns=['season', 'games', 'player_id', 'position']
            )
            max_season = int(existing['season'].max())
            # Rebuild if parquet does not include the inference season
            needs_rebuild = max_season < cur_season
            if not needs_rebuild:
                cur = existing.loc[existing['season'] == cur_season]
                # Stale builds used offseason week=22 as games and/or appended sparse weekly dupes
                stale_games = (not cur.empty) and (cur['games'].median() > 18)
                stale_dupes = (not cur.empty) and cur.duplicated(
                    subset=['player_id', 'position']
                ).any()
                needs_rebuild = stale_games or stale_dupes
            if needs_rebuild:
                logging.info(
                    f"Rebuilding data.parquet (max_season={max_season}, cur_season={cur_season})"
                )

        if needs_rebuild:
            fdf = FantasyDataFrame(cur_season=cur_season)
            self.fdf = fdf
            logging.info("Creating parquet...")
            fdf.players_stats.to_parquet(parquet_path, index=False)

        self.rb_seasonal = Seasonal(points_type='ppr', position = 'RB', type = 'xgb', cur_season = cur_season)
        # self.qb_seasonal = Seasonal(points_type='ppr', position = 'QB', type = 'xgb', cur_season = cur_season)
        # self.te_seasonal = Seasonal(points_type='ppr', position = 'TE', type = 'xgb', cur_season = cur_season)
        # self.wr_seasonal = Seasonal(points_type='ppr', position = 'WR', type = 'xgb', cur_season = cur_season)


    def run(self):
        rb_seasonal = self.rb_seasonal
        rb_seasonal.corr()

        # only outputting standard/game right now
        rb_seasonal.set_features()
        rb_seasonal.train_model(rb_seasonal.model)
        rb_seasonal.test_model()
        rb_seasonal.cross_validate()

        
        # qb_seasonal = self.qb_seasonal
        # qb_seasonal.corr()

        # # only outputting standard/game right now
        # # rb_seasonal.set_features()
        # qb_seasonal.train_model(qb_seasonal.model)
        # qb_seasonal.test_model()
        # qb_seasonal.cross_validate()

        # te_seasonal = self.te_seasonal
        # te_seasonal.corr()

        # # only outputting standard/game right now
        # te_seasonal.set_features()
        # te_seasonal.train_model(te_seasonal.model)
        # te_seasonal.test_model()
        # te_seasonal.cross_validate()

        # wr_seasonal = self.wr_seasonal
        # wr_seasonal.corr()

        # # only outputting standard/game right now
        # # wr_seasonal.set_features()
        # wr_seasonal.train_model(wr_seasonal.model)
        # wr_seasonal.test_model()
        # wr_seasonal.cross_validate()



        print(rb_seasonal)
        # print(te_seasonal)
        # print(wr_seasonal)
        # print(qb_seasonal)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current_season", type=int, default=nfl.get_current_season())
    args = parser.parse_args()


    logging.basicConfig(level=logging.INFO)
    ModelExecutor(args.current_season).run()