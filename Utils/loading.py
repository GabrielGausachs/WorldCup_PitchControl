import os
import json
import pandas as pd
from kloppy import pff
from databallpy import get_game_from_kloppy

def load_events_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.json_normalize(data)

def load_files(base_path, game_id=None):
    efile_path = os.path.join(base_path, "eventdata", f"{game_id}.json")
    event_df = load_events_from_json(efile_path)
    print("-" * 50)
    print(f"Loaded {game_id}: {event_df.shape[0]} events")
        
    tfile = os.path.join(base_path, "trackingdata_parquet", f"{game_id}.parquet")
    tracking_df = pd.read_parquet(tfile)
    print(f"Loaded {game_id}: {tracking_df.shape[0]} rows")
    
    return event_df, tracking_df

def load_game_from_pff(base_path, game_id):
    dataset = pff.load_tracking(
        meta_data= os.path.join(base_path,"metadata",f"{game_id}.json"),
        roster_meta_data= os.path.join(base_path,"rosters",f"{game_id}.json"),
        raw_data= os.path.join(base_path,"trackingdata_clean",f"{game_id}_clean.jsonl.bz2"),
        coordinates="pff",
        only_alive=False
    )
    dataset.metadata.frame_rate = int(dataset.metadata.frame_rate)
    game = get_game_from_kloppy(tracking_dataset=dataset)
    game.tracking_data.filter_tracking_data(column_ids="ball", filter_type="savitzky_golay", window_length=25, polyorder=2)
    game.tracking_data.add_velocity(game.get_column_ids() + ["ball"], allow_overwrite=True)
    game.tracking_data.add_acceleration(game.get_column_ids() + ["ball"], allow_overwrite=True)
    print(f"Loaded tracking data for game {game_id}")
    return game