import numpy as np
from scipy.interpolate import interp1d
from osrparse import Replay
from osrparse.utils import ReplayEventOsu, Key
import hashlib
from datetime import datetime

"""
Position model predictions are done in 60hz
This function will interpolate the points so that there is position update every 8ms
It looks better but is not necessary
"""
def _interpolate_position(positions):
    time = positions[:, 0]
    x = positions[:, 1]
    y = positions[:, 2]

    dt = np.diff(time)
    gap_threshold = 25  # milliseconds
    gap_indices = np.where(dt > gap_threshold)[0]

    # Define segment start and end indices
    segment_starts = np.insert(gap_indices + 1, 0, 0)
    segment_ends = np.append(gap_indices, len(time) - 1)

    interpolated_time = []
    interpolated_x = []
    interpolated_y = []

    # Desired sampling interval (approximate 120Hz)
    sampling_interval = 8  # milliseconds

    for start_idx, end_idx in zip(segment_starts, segment_ends):
        segment_time = time[start_idx:end_idx + 1]
        segment_x = x[start_idx:end_idx + 1]
        segment_y = y[start_idx:end_idx + 1]

        # Rounding segment times to integers
        segment_start_time = int(np.ceil(segment_time[0]))
        segment_end_time = int(np.floor(segment_time[-1]))

        # Creating new integer time points at the desired sampling interval
        new_time = np.arange(segment_start_time, segment_end_time + 1, sampling_interval)
        
        # No new time points in this segment
        if len(new_time) == 0:
            continue  

        kind = 'cubic' if len(segment_time) >= 4 else 'linear'

        # Interpolating x and y values
        x_interp = interp1d(segment_time, segment_x, kind=kind)
        y_interp = interp1d(segment_time, segment_y, kind=kind)

        new_x = x_interp(new_time)
        new_y = y_interp(new_time)

        # Collecting interpolated data
        interpolated_time.append(new_time)
        interpolated_x.append(new_x)
        interpolated_y.append(new_y)

    # Combining all segments
    final_time = np.concatenate(interpolated_time)
    final_x = np.concatenate(interpolated_x)
    final_y = np.concatenate(interpolated_y)

    # Combining into final data array
    final_data = np.column_stack((final_time, final_x, final_y))
    return final_data

"""
Function to merge the position and keypress prediction
"""
def _merge_pos_key(positions_matrix, keypresses_matrix):
    times_array = positions_matrix[:, 0]
    update_indices = []
    
    # Function to find the index of the time in positions_matrix closest to target_time
    def find_nearest_index(times_array, target_time):
        idx = np.searchsorted(times_array, target_time)
        if idx == 0:
            return 0
        elif idx == len(times_array):
            return len(times_array) - 1
        else:
            prev_time = times_array[idx - 1]
            next_time = times_array[idx]
            if abs(prev_time - target_time) < abs(next_time - target_time):
                return idx - 1
            else:
                return idx
    
    # Iterate over each keypress event
    for keypress in keypresses_matrix:
        keypress_time, end_time, keycode = keypress
        keypress_time = float(keypress_time)
        end_time = float(end_time)
        keycode = int(keycode)
        
        if end_time == -1:
            # Single keypress event, find the nearest index
            idx = find_nearest_index(times_array, keypress_time)
            update_indices.append((idx, keycode))
        else:
            # Finding indices in positions_matrix within the time range
            idx_start = np.searchsorted(times_array, keypress_time, side='left')
            idx_end = np.searchsorted(times_array, end_time, side='right')
            
            if idx_end > idx_start:
                # Updating keycodes in the time range
                for idx in range(idx_start, idx_end):
                    update_indices.append((idx, keycode))
            else:
                # No positions within time range, find the nearest index to keypress_time
                idx = find_nearest_index(times_array, keypress_time)
                update_indices.append((idx, keycode))
    
    # Removing duplicate indices to avoid redundant updates
    update_indices = list(set(update_indices))
    
    # Update the keycodes in positions_matrix
    for idx, keycode in update_indices:
        positions_matrix[idx, 3] = keycode
    
    return positions_matrix

"""
Function to process position and keypress prediction into format used to generate replay
"""
def post_process(p_pred, k_pred, p_time, k_time, k_end_time):
    
    # Flattening tensors and converting to np arrays
    positions = [tensor.flatten().tolist() for tensor in p_pred]
    positions = np.array(positions)
    keypresses = [tensor.flatten().tolist() for tensor in k_pred]
    keypresses = np.array(keypresses)
    
    # Reverting normalization on positions
    positions[:, 0] *= 512
    positions[:, 1] *= 384
    
    # Creating time column of time of each prediction
    # Column is added to predictions to form a matrix
    pos_time_column = np.array(p_time).astype(int).reshape(-1, 1)
    pos_matrix = np.concatenate((pos_time_column, positions), axis=1)
    
    # Keypress model outputs are indices 0 or 1
    # 0 is used for no keypress so increment all predictions
    keypresses = keypresses + 1
    key_time_column = np.array(k_time).astype(int).reshape(-1, 1)
    key_end_time_column = np.array(k_end_time).astype(int).reshape(-1, 1)
    key_matrix = np.concatenate((key_time_column, key_end_time_column, keypresses), axis=1)
    
    # Interpolating positions
    pos_matrix_interp = _interpolate_position(pos_matrix)
    
    # Appending a column of 0s for keypresses to positions matrix
    # This column will be updated during merging
    num_rows = pos_matrix_interp.shape[0]
    dummy_keys_column = np.zeros((num_rows, 1))
    pos_matrix_interp = np.concatenate((pos_matrix_interp, dummy_keys_column), axis=1)
    
    # Merging the position and keypress predictions and forming a matrix suitable for replay
    replay_matrix = _merge_pos_key(pos_matrix_interp, key_matrix)
    
    return replay_matrix

"""
Function to determine beatmap md5 hash and name
"""
def _get_beatmap_name_hash(map_path, max_song_len=30, max_diff_len=30):
    with open(map_path, 'r', encoding='utf-8') as file:
        lines = file.read()

    metadata = lines.split('\n\n')[3]
    content = metadata.split('\n')
    song_name, diff_name = content[1].split(':')[1], content[6].split(':')[1]
    song_name = song_name[:max_song_len]
    diff_name = diff_name[:max_diff_len]
    map_name = f'{song_name}[{diff_name}]'
    
    hash_data = lines.replace('\r\n', '\n').replace('\n', '\r\n')
    hash_md5 = hashlib.md5(hash_data.encode('utf-8')).hexdigest()
    
    return map_name, hash_md5

"""
Function to save the result from post_process into .osr format
"""
def save_replay(replay_predictions, replay_path, map_path, pos_name, key_name):
    replay = Replay.from_path(replay_path)
    map_name, map_hash = _get_beatmap_name_hash(map_path)

    time_deltas = np.diff(replay_predictions[:, 0], prepend=replay_predictions[0, 0])
    time_deltas[0] = replay_predictions[0, 0]

    keymap = [0, 1, 2, 11]
    replay_data = []
    for i in range(len(replay_predictions)):
        pred = replay_predictions[i]
        key = Key(keymap[int(pred[3])])
        replay_event = ReplayEventOsu(int(time_deltas[i]), float(pred[1]), float(pred[2]), key)
        replay_data.append(replay_event)
        
    markers = [
        ReplayEventOsu(0, 256.0, -500.0, Key(0)),
        ReplayEventOsu(-1, 256.0, -500.0, Key(0))
    ]

    replay.replay_data = markers + replay_data
    
    pos_name, key_name = pos_name.split(']')[1], key_name.split(']')[1]
    if pos_name == key_name:
        replay.username = pos_name
    else:
        replay.username = f'{pos_name}_{key_name}'
    replay.beatmap_hash = map_hash
    replay.timestamp = datetime.now()

    replay.write_path(f"./{replay.username}-{map_name}.osr")