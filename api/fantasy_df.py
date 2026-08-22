import pandas as pd
import nflreadpy as nfl
import numpy as np
import pyarrow
import logging
import gc


class FantasyDataFrame:
    def __init__(self, summary_level = 'reg', cur_season = 2026):
        logging.info('Initializing...')
        # normalize to football year season
        years = [year for year in range(2016, cur_season+1)]
        self.cur_season = cur_season
        self.years = years
        self.pos = ['TE', 'RB', 'WR', 'QB']
        self.summary_level = summary_level
        self.load_data()

    def load_data(self):

        # use players_stats as base file
        players_stats = nfl.load_player_stats(self.years, self.summary_level).to_pandas()
        players_stats = players_stats[players_stats['season_type']=='REG']
        players_stats=players_stats.drop_duplicates(subset = ['player_id', 'position', 'season'])

        logging.info(f"Number of rows: {players_stats.shape[0]}")
        
        self.players_stats = players_stats
        self.map_ids()
        players_stats = self.players_stats
        logging.info(f"Number of rows: {players_stats.shape[0]}")

        # Mid-season only: seasonal rollups may be incomplete, so backfill cur_season from weekly.
        # Do not append when cur_season is already present in seasonal data — that creates sparse duplicate rows.
        current_week = nfl.get_current_week()
        season_already_loaded = self.cur_season in set(players_stats['season'].dropna().unique())
        in_season = 1 <= current_week <= 18
        # #region agent log
        try:
            import json, time
            cur_miss = {}
            if self.cur_season in set(players_stats['season'].dropna().unique()):
                cur = players_stats.loc[players_stats['season']==self.cur_season]
                for c in ['carries','targets','receptions','rushing_yards','fantasy_points_ppr']:
                    if c in cur.columns:
                        cur_miss[c] = float(cur[c].isna().mean())
            open("/Users/evanratliff/Documents/nfl-project/ml-vs-human-analysts-backend/.cursor/debug-9fea92.log","a").write(json.dumps({"sessionId":"9fea92","runId":"post-fix","hypothesisId":"A","location":"fantasy_df.py:overlay_gate","message":"weekly overlay gate","data":{"cur_season":int(self.cur_season),"current_week":int(current_week),"in_season":bool(in_season),"season_already_loaded":bool(season_already_loaded),"will_overlay":bool(in_season and not season_already_loaded),"pre_overlay_cur_missing":cur_miss,"pre_overlay_ncols":int(players_stats.shape[1]),"pre_overlay_nrows":int(players_stats.shape[0])},"timestamp":int(time.time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion
        if in_season and not season_already_loaded:
            players_weekly_stats = nfl.load_player_stats(self.cur_season, 'week').to_pandas()
            players_weekly_stats = players_weekly_stats.loc[players_weekly_stats['week'] <= 18]

            skip_sum = {'week', 'season'}
            numeric_cols = players_weekly_stats.select_dtypes(include=[np.number]).columns
            agg_dict = {col: 'first' for col in players_weekly_stats.columns}
            agg_dict.update({col: 'sum' for col in numeric_cols if col not in skip_sum})
            players_weekly_stats = players_weekly_stats.groupby(['player_id'], as_index=False).agg(agg_dict)

            players_weekly_stats['fantasy_points_half_ppr'] = players_weekly_stats['fantasy_points_ppr'] - players_weekly_stats['receptions'] * 0.5
            players_weekly_stats['fantasy_points_standard'] = players_weekly_stats['fantasy_points_ppr'] - players_weekly_stats['receptions'] * 1
            players_weekly_stats['season'] = self.cur_season
            if 'recent_team' in players_weekly_stats.columns and (
                'team' not in players_weekly_stats.columns or players_weekly_stats['team'].isna().all()
            ):
                players_weekly_stats['team'] = players_weekly_stats['recent_team']
            players_weekly_stats = players_weekly_stats.drop_duplicates(subset=['season', 'player_id'])
            logging.info(list(players_weekly_stats.columns))

            players_stats = pd.concat(
                [players_stats, players_weekly_stats],
                ignore_index=True,
                sort=False,
            )
            # #region agent log
            try:
                import json, time
                cur = players_stats.loc[players_stats['season']==self.cur_season]
                miss = {c: float(cur[c].isna().mean()) for c in ['carries','targets','receptions','rushing_yards','fantasy_points_ppr'] if c in cur.columns}
                open("/Users/evanratliff/Documents/nfl-project/ml-vs-human-analysts-backend/.cursor/debug-9fea92.log","a").write(json.dumps({"sessionId":"9fea92","runId":"post-fix","hypothesisId":"A","location":"fantasy_df.py:after_overlay","message":"current season missingness after sparse weekly concat","data":{"cur_season":int(self.cur_season),"cur_rows":int(len(cur)),"concat_cols":['player_id','season','fantasy_points_half_ppr','fantasy_points_ppr','fantasy_points_standard','player_name','position','headshot_url'],"miss":miss},"timestamp":int(time.time()*1000)})+"\n")
            except Exception:
                pass
            # #endregion
        else:
            logging.info(
                f"Skipping weekly overlay for {self.cur_season} "
                f"(in_season={in_season}, season_already_loaded={season_already_loaded})"
            )

        players_stats['fantasy_points_standard'] = players_stats['fantasy_points_ppr']-(players_stats['receptions']*1)
        players_stats['fantasy_points_half_ppr'] = players_stats['fantasy_points_ppr']-(players_stats['receptions']*0.5)

        logging.info('Importing games spent on IR...')
        # track games played and games missed on IR
        injuries = nfl.load_injuries(self.years)[['gsis_id', 'season', 'week','report_status', 'position']].to_pandas()

        # count games on ir and games played
        print(injuries)
        injuries=injuries.loc[((injuries['week'] <=17) & (injuries['season']>=2021)) | ((injuries['week'] <=16) & (injuries['season']<2021))]
        injuries['out_games'] = injuries.groupby(['gsis_id','season'])['report_status'].transform(lambda x: (x == 'Out').sum())
        injuries['total_missed_games'] = injuries['out_games']
        injuries.drop_duplicates(subset = ['gsis_id', 'season'], inplace=True)

        players_stats = players_stats.merge(
            injuries[['gsis_id', 'position', 'season', 'total_missed_games']],
            left_on = ['player_id', 'position', 'season'],
            right_on = ['gsis_id', 'position', 'season'],
            how = 'left'   
        )

        logging.info(f"Number of rows: {players_stats.shape[0]}")


        del injuries
        gc.collect()
        # Fill NaN with 0 and convert to int
        players_stats['total_missed_games'] = players_stats['total_missed_games'].fillna(0).astype(int)

        # Prefer nflverse games played. Season-length minus IR made current-season
        # games a constant (IR often unavailable) and diluted per-game features.
        current_season = self.cur_season
        current_week = nfl.get_current_week()
        cur_season_games = current_week if 1 <= current_week <= 18 else 17
        fallback_games = np.select(
            [
                players_stats['season'] == current_season,
                players_stats['season'].between(2021, current_season - 1),
                players_stats['season'] < 2021,
            ],
            [
                cur_season_games,
                17,
                16,
            ],
            default=17,
        )
        fallback_games = pd.Series(fallback_games, index=players_stats.index) - players_stats['total_missed_games']
        played = pd.to_numeric(players_stats['games'], errors='coerce') if 'games' in players_stats.columns else None
        if played is not None:
            players_stats['games'] = played.fillna(fallback_games)
        else:
            players_stats['games'] = fallback_games
        players_stats['games'] = players_stats['games'].clip(lower=1)


        logging.info("Importing Next Gen Stats...")

        # track next gen stats
        # receiving
        ngs_receiving = nfl.load_nextgen_stats([year for year in self.years if year >=2016], 'receiving').to_pandas()
        ngs_receiving.drop_duplicates(subset=['player_gsis_id', 'season'], inplace=True)
        players_stats = players_stats.merge(
            ngs_receiving[['player_gsis_id', 'season', 'avg_cushion', 'avg_separation', 'percent_share_of_intended_air_yards', 'catch_percentage', 'avg_yac', 'avg_expected_yac', 'avg_yac_above_expectation']],
            left_on=['player_id', 'season'],
            right_on=['player_gsis_id', 'season'],
            how='left'
        ).drop(columns=['player_gsis_id'])

        del ngs_receiving
        gc.collect()
        logging.info(f"Number of rows: {players_stats.shape[0]}")



        # rushing
        ngs_rushing = nfl.load_nextgen_stats([year for year in self.years if year >=2016], 'rushing').to_pandas()
        ngs_rushing.drop_duplicates(subset=['player_gsis_id', 'season'], inplace=True)
        players_stats = players_stats.merge(
            ngs_rushing[['player_gsis_id', 'season', 'efficiency', 'percent_attempts_gte_eight_defenders', 'avg_time_to_los', 'expected_rush_yards', 'rush_yards_over_expected', 'rush_pct_over_expected', 'rush_yards_over_expected_per_att']],
            left_on=['player_id', 'season'],
            right_on=['player_gsis_id', 'season'],
            how='left'
        )

        del ngs_rushing
        gc.collect()

        # passing
        ngs_passing = nfl.load_nextgen_stats([year for year in self.years if year >=2016], 'passing').to_pandas()
        ngs_passing.drop_duplicates(subset=['player_gsis_id', 'season'], inplace=True)

        players_stats = players_stats.merge(
            ngs_passing[['player_gsis_id', 'season', 'avg_air_distance', 'max_air_distance', 'avg_time_to_throw', 'avg_completed_air_yards', 'avg_intended_air_yards', 'avg_air_yards_differential', 'aggressiveness', 'max_completed_air_distance', 'avg_air_yards_to_sticks', 'passer_rating', 'completion_percentage', 'expected_completion_percentage', 'completion_percentage_above_expectation', 'avg_air_distance', ]],
            left_on=['player_id', 'season'],
            right_on=['player_gsis_id', 'season'],
            how='left'
        )

        del ngs_passing
        gc.collect()

        # create per-___ stats
        players_stats['pass_tds/game'] = players_stats['passing_tds']/players_stats['games']
        players_stats['pass_air_yards/game'] = players_stats['passing_air_yards']/players_stats['games']
        players_stats['carries/game'] = players_stats['carries']/players_stats['games']
        players_stats['yards/carry'] = players_stats['rushing_yards']/players_stats['carries']
        players_stats['rushing_tds/game'] = players_stats['rushing_tds']/players_stats['games']
        players_stats['receiving_tds/game'] = players_stats['receiving_tds']/players_stats['games']
        players_stats['turnovers/game'] = (players_stats['rushing_fumbles'] + players_stats['sack_fumbles']+players_stats['passing_interceptions'] )/players_stats['games']
        players_stats['adot'] = players_stats['passing_air_yards']/players_stats['attempts']
        players_stats['targets_game'] = players_stats['targets']/players_stats['games']

        logging.info("Importing players_stats roster data...")
        # import roster data
        # players_stats['team'] = ""
        #TODO: keep cleaning data from here. also group stats for this model by season and gsis_id
        rosters = nfl.load_rosters(self.years).to_pandas()
        rosters.drop_duplicates(subset=['gsis_id', 'season'], inplace=True)
        players_stats = players_stats.merge(
            rosters[['gsis_id', 'season', 'height', 'weight', 'college', 'status', 'years_exp']],
            left_on=['player_id', 'season'],
            right_on=['gsis_id', 'season'],
            how='left'
        )

        logging.info(f"Number of rows: {players_stats.shape[0]}")


        del rosters

        gc.collect()

        # schedule = nfl.load_schedule(self.years).to_pandas()


        # since data source has changed, use players_stats roster's depth_chart_position for now. 
        # import position on depth chart
        # depth_charts = nfl.load_depth_charts(self.years).to_pandas()

        # # depth_charts = depth_charts.loc[depth_charts['week']==1.0]
        # players_stats = players_stats.merge(
        #     depth_charts[['gsis_id', 'week', 'depth_team']],
        #     left_on = ['player_gsis_id', 'week'],
        #     right_on= ['gsis_id', 'season'],
        #     how='left',
        #     suffixes = ["", '']
        # )

        # drop because NaN causes error for astype
        # players_stats.dropna(subset = ['depth_chart_position'], inplace = True)
        # players_stats['depth_chart_position'] = players_stats['depth_chart_position'].astype('int')

        # will assign win totals based on team column (might need heavy mapping w dicts)
        #https://www.nfeloapp.com/nfl-power-ratings/nfl-win-totals/
        logging.info("Importing Vegas Win Total Lines...")
        win_totals = pd.read_csv('nfl-win-totals-2026.csv')[['Season','Team','Adj. Total']]
        win_totals['Season'] = win_totals['Season'].astype('int')
        win_totals['Team'] = win_totals['Team'].astype('str')
        win_totals['Adj. Total'] = win_totals['Adj. Total'].astype('float')

        win_totals['Team'] = win_totals['Team'].str.strip()
        # players_stats['team'] = players_stats['team'].str.strip()
        win_totals.drop_duplicates(subset=['Team', 'Season'], inplace=True)

        players_stats = players_stats.merge(
            win_totals[['Team', 'Season', 'Adj. Total']],
            left_on=['team', 'season'],
            right_on =['Team', 'Season'],
            how='left'
        ).rename(columns={'Adj. Total': 'win_total_adj_line'})

        logging.info(f"Number of rows: {players_stats.shape[0]}")


        del win_totals
        gc.collect()

        ## rookies
        logging.info("Importing draft data...")
        # import rec_yards, rec_tds, etc. college production
        draft_picks = nfl.load_draft_picks(self.years).to_pandas()

        draft_picks['draft_age'] = draft_picks['age']
        draft_picks['career_games'] = draft_picks['games']

        draft_picks.drop_duplicates(subset=['gsis_id'], inplace=True)

        # Merge draft_picks into players_stats based on the player_id and gsis_id columns
        players_stats = players_stats.merge(
            draft_picks[['gsis_id', 'pick', 'allpro', 'draft_age', 'w_av', 'car_av', 'dr_av', 'career_games']], 
            left_on='player_id',  
            right_on='gsis_id',   
            how='left'            
        )
        players_stats['age'] = players_stats['draft_age'] + players_stats['years_exp']

        del draft_picks
        gc.collect()

        logging.info("Importing combine data...")

        # get combine stats: for RBs, WRs, TEs, and map by player_name and pos
        combine = nfl.load_combine(self.years).to_pandas()

        combine = combine.rename(columns={'pos': 'position'})
        combine['player_name'] = combine['player_name'].str.strip()
        combine['position'] = combine['position'].str.strip()
        combine.drop_duplicates(subset=['player_name', 'position', 'draft_ovr'], inplace=True)


        players_stats = players_stats.merge(
            combine[['player_name', 'position', 'draft_ovr', 'forty', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']], 
            left_on = ['player_name', 'position', 'pick'],
            right_on=['player_name', 'position','draft_ovr' ], 
            how='left'
        )

        del combine
        gc.collect()

        logging.info('Removing duplicate columns')
        ## TODO: support wopr_x, wopr_y
        # drop duplicate _x columns created by .merge
        # columns = set()
        # for column in players_stats.columns:
        #     columns.add(column)
        #     if ((column[-2:] == '_x') & (column != 'wopr_x')) or column in columns:
        #         players_stats.drop(column, axis=1, inplace=True)
        # # delete the _y created from the .merge function
        #     if (column[-2:] == '_y') & (column != 'wopr_y'):
        #         players_stats.rename(columns={column:column[:-2]}, inplace=True)


        logging.info('Creating future values (ground truth) for training and testing')

        players_stats['next_season'] = players_stats['season']-1

        ## assign future stats for y values
        #important: get future team to load other_epas
        players_stats = players_stats.merge(
            players_stats[['player_id', 'next_season', 'fantasy_points','fantasy_points_ppr', 'fantasy_points_half_ppr','games', 'team']],
            left_on = ['player_id', 'season'],
            right_on = ['player_id', 'next_season'],
            how = 'left',
            suffixes = ('', '_future')                                                                                                      
        )

        players_stats['future_stardard/game'] = players_stats['fantasy_points_future']/players_stats['games_future']
        players_stats['future_ppr/game'] = players_stats['fantasy_points_ppr_future']/players_stats['games_future']
        players_stats['future_half_ppr/game'] = players_stats['fantasy_points_half_ppr_future']/players_stats['games_future']

        players_stats['standard/game'] = players_stats['fantasy_points_future']/players_stats['games']
        players_stats['ppr/game'] = players_stats['fantasy_points_ppr']/players_stats['games']
        players_stats['half_ppr/game'] = players_stats['fantasy_points_half_ppr']/players_stats['games']

        # each position model class will need to subtract their own epa to get the result, other_rbs_epa
        logging.info("Importing player contextual stats...")
        # --- 1. Define position configuration ---
        position_config = {
            'team_qbs': {
                'sum_cols': ['rushing_tds'],
                'mean_cols': ['aggressiveness','pick', 'years_exp', 'turnovers/game', 'carries/game', 
                            'completion_percentage_above_expectation', 'passing_epa'],
                'position_filter': 'QB'
            },
            'team_wrs': {
                'sum_cols': ['receiving_tds','air_yards_share'],
                'mean_cols': ['avg_cushion', 'avg_separation', 'avg_yac_above_expectation','catch_percentage', 
                            'height', 'receiving_epa', 'pick'],
                'position_filter': 'WR'
            },
            'team_rbs': {
                'sum_cols': ['receiving_tds', 'carries', 'receiving_yards', 'air_yards_share'],
                'mean_cols': ['years_exp', 'age', 'weight', 'percent_attempts_gte_eight_defenders',
                            'avg_time_to_los', 'rush_yards_over_expected_per_att', 'rush_pct_over_expected', 
                            'receiving_epa', 'rushing_epa', 'pick'],
                'position_filter': 'RB'
            },
            'team_tes': {
                'sum_cols': ['receiving_tds', 'receiving_yards', 'air_yards_share'],
                'mean_cols': ['catch_percentage', 'wopr', 'percent_share_of_intended_air_yards', 'receiving_epa'],
                'position_filter': 'TE'
            }
        }

        # --- 2. Helper: build aggregation dictionary ---
        def build_agg_dict(sum_cols, mean_cols, suffix):
            agg = {}
            for c in sum_cols:
                agg[f"{c}_{suffix}_sum"] = (c, 'sum')
            for c in mean_cols:
                agg[f"{c}_{suffix}_mean"] = (c, 'mean')
            agg[f"num_{suffix}"] = ('player_id', 'count')
            return agg

        # --- 3. Pre-filter DataFrame ---
        logging.info("Dropping np.NaN rows")
        players_stats = players_stats.dropna(subset=['position', 'season'])
        logging.info(f"Number of rows: {players_stats.shape[0]}")

        # --- 4. Aggregate all positions ---
        logging.info("Aggregating all positional team_aggs - perform .loc, build aggregation dict, and .groupby team and season")
        team_agg_list = []
        for suffix, cfg in position_config.items():
            pos = cfg['position_filter']
            df_pos = players_stats.loc[players_stats['position'] == pos]
            agg_dict = build_agg_dict(cfg['sum_cols'], cfg['mean_cols'], suffix)
            team_agg = df_pos.groupby(['season','team']).agg(**agg_dict).reset_index()
            team_agg_list.append(team_agg)

        # Merge back into players_stats once
        logging.info("Merging team aggs into players_stats")
        # add one at a time for memory constraint reasons
        for i, df in enumerate(team_agg_list):
            logging.info(f"Merging df #{i} {df.shape[0]}")
            df.drop_duplicates(subset = ['season','team'], inplace=True)
            # merge onto future team in order to have up to date contexual data for every player
            # players_stats = players_stats.merge(df, left_on=['next_season_future','team_future'], right_on = ['season', 'team'], how='left')
            players_stats = players_stats.merge(df, on= ['season', 'team'], how='left')

        logging.info("Computing self-exclusive team quality statistics for all players")
        for team_suffix, position_stats in position_config.items():
            num_players_in_team = f"num_{team_suffix}"
            logging.info(f"Computing sums for {team_suffix}, {position_stats}")
            
            for stat_name in position_stats['sum_cols']:
                team_stat_total = f"{stat_name}_{team_suffix}_sum"       # total sum for the team
                stat_excluding_self = f"{stat_name}_other_{team_suffix}" # total minus this player's value
                if team_stat_total in players_stats.columns:
                    players_stats[stat_excluding_self] = players_stats[team_stat_total] - players_stats[stat_name].fillna(0)

            logging.info(f"Computing means for {team_suffix}, {position_stats}")
            # Mean-based columns: ((count * team_mean) - own) / (count-1)
            for stat_name in position_stats['mean_cols']:
                team_stat_mean = f"{stat_name}_{team_suffix}_mean"       # average for the team
                stat_excluding_self = f"{stat_name}_other_{team_suffix}" # mean excluding this player
                if team_stat_mean in players_stats.columns:
                    numerator = (players_stats[num_players_in_team] * players_stats[team_stat_mean]) - players_stats[stat_name].fillna(0)
                    denominator = players_stats[num_players_in_team] - 1
                    players_stats[stat_excluding_self] = np.where(denominator > 0, numerator / denominator, np.nan)

        logging.info("dropping duplicate columns")
        logging.info(list(players_stats.columns))
        # transpose for columns --> rows, drop duplicates, then rows --> columns
        columns = set(list(players_stats.columns))
        for col in list(players_stats.columns):
            if col in columns:
                columns.remove(col)
            else:
                players_stats.drop(col, axis = 1, inplace=True)
        
        players_stats.to_csv('players_stats.csv', index=False)
        
        self.players_stats = players_stats

    def map_ids(self):
        df = self.players_stats
        ## map all values in column id to names
        # import df of columns 'id', 'name', 'position'
        mappings = nfl.load_ff_playerids().to_pandas()[['name', 'position', 'gsis_id', 'twitter_username', 'team']]
        mappings.set_index('gsis_id', inplace=True)

        # for dict transformation eliminate duplicate key-value pairs
        mappings.drop_duplicates(inplace=True)

        # df to dict for mapping
        mappings_name_dict = mappings['name'].to_dict()
        mappings_pos_dict = mappings['position'].to_dict()
        mappings_twitter_dict = mappings['twitter_username'].to_dict()
        mappings_team_dict = mappings['team'].to_dict()

        # use .map to map flat dictionary, "name+position" -->id
        logging.info("Mapping IDs...")
        df['player_name'] = df['player_id'].map(mappings_name_dict)
        df['position'] = df['player_id'].map(mappings_pos_dict)
        df['twitter_username'] = df['player_id'].map(mappings_twitter_dict)

        # Prefer season-specific nflverse team. ff_playerids.team is the *current*
        # roster (mostly FA for retired players) and would inflate historical team aggs.
        team_aliases = {
            'NOS': 'NO', 'NEP': 'NE', 'SFO': 'SF', 'TBB': 'TB', 'KCC': 'KC',
            'GBP': 'GB', 'LVR': 'LV', 'JAC': 'JAX', 'WSH': 'WAS', 'GNB': 'GB',
            'NOR': 'NO', 'NWE': 'NE', 'TAM': 'TB', 'KAN': 'KC', 'LAS': 'LV',
            'OAK': 'LV', 'SD': 'LAC', 'STL': 'LAR',
        }
        mapped_team = df['player_id'].map(mappings_team_dict).replace(team_aliases)
        if 'recent_team' in df.columns:
            season_team = df['recent_team']
        elif 'team' in df.columns:
            season_team = df['team']
        else:
            season_team = None
        if season_team is not None:
            df['team'] = season_team.replace(team_aliases).fillna(mapped_team)
        else:
            df['team'] = mapped_team

        # drop unmapped-id players from df (fantasy defenses, other edgecases)
        df.dropna(subset=['position'], inplace=True)
        self.players_stats = df

