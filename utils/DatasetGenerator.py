from utils.DataParser import parse_map, parse_replay_pos, parse_replay_key
import pandas as pd
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
import numpy as np

W = 512
H = 384
    
def get_X(folder_path, hr, file_path=None):
    if file_path:
        _, feats = parse_map(file_path, hr)
    else:
        _, feats = parse_map(list(Path(folder_path).glob('*.osu'))[0], hr)
    
    x = pd.DataFrame(feats, columns=[
        'x', 'y', 'time', 'type', 'end_time'
    ])
    
    # x = x.sort_values('time')
    x['delta_time'] = (x['time'].diff()).fillna(0)
    # x = x.drop(['end_time'], axis=1)
    
    x['x_norm'] = x['x'] / W
    x['y_norm'] = x['y'] / H
    types = ['1.0', '6.0', '7.0', '12.0', '13.0']
    type_ohe = pd.get_dummies(x['type'], prefix='type').reindex(columns=[f'type_{t}' for t in types], fill_value=0)
    x = pd.concat([x, type_ohe], axis=1)
    x = x.drop(['x', 'y', 'type'], axis=1)
    
    return x

def get_y(path, type):
    if type == 'pos':
        target, hr = parse_replay_pos(list(Path(path).glob('*.osr'))[0])
    elif type == 'key':
        target = parse_replay_key(list(Path(path).glob('*.osr'))[0])
        hr = False

    y = pd.DataFrame(target, columns=[
        'x', 'y', 'time', 'keycode'
    ])
    y['delta_time'] = (y['time'].diff()).fillna(0)
    y['x_norm'] = y['x'] / W
    y['y_norm'] = y['y'] / H
    
    # Compressing keycodes where different keycode refer to same action
    y['keycode'] = y['keycode'].replace({
        5: 1,   #key 1
        2: 10,  #key 2
        15: 11, #key 1 + 2
        21: 0   #smoke
    })
    
    # Separating into position targets
    y_pos = y[['x_norm', 'y_norm', 'delta_time', 'time']].copy()
    
    # Separating into keypress targets
    y_key = y[['keycode', 'time']].copy()

    # # Define the keycodes  
    # keycodes = ['0.0', '1.0', '10.0', '11.0']

    # # One-hot encode the 'keycode' column 
    # keycode_ohe = pd.get_dummies(y_key['keycode'], prefix='key').reindex(
    #     columns=[f'key_{k}' for k in keycodes], fill_value=0)

    # # Combine 'time' and the one-hot encoded keycodes
    # y_key = pd.concat([y_key[['time']], keycode_ohe], axis=1)
    
    if type == 'pos':
        return y_pos, hr
    elif type == 'key': 
        return y_key, hr

def _get_Xy(path, type):
    df_y, hr = get_y(path, type)
    df_X = get_X(path, hr)
    return [df_X, df_y, path]
    
# IDEA lets do a separate seqeunce generation function for keypresses
# Xi[sequence of hit objects]
# yi[sequence of non consequtive keypresses] the len of yi should be proportional to len of Xi (except for slider ticks) since each hitobject requires a key down and up event
# probably need 2 models since the inputs will not be related

def _remove_overlap_keypresses(keypresses):
    transformed_arr = keypresses.flatten()
    
    # Initialize last_key as None
    last_key = None
    
    for i in range(len(transformed_arr)):
        current = transformed_arr[i]
        
        if current == 11.0:
            if last_key == 1.0:
                transformed_arr[i] = 10.0
            elif last_key == 10.0:
                transformed_arr[i] = 1.0
            else:
                # No preceding keycode; decide on default behavior
                # Here, we'll choose to leave it as 11.0
                # Alternatively, you could set a default like 1.0 or 10.0
                transformed_arr[i] = 1.0
        else:
            # Update last_key if the current keycode is not 11.0 and not 0.0
            if current in [1.0, 10.0]:
                last_key = current
            # If current is 0.0, we do not update last_key
            
    # OHE
    # Find unique categories and sort them
    # unique_categories = np.unique(transformed_arr)
    
    # # Create a mapping from category to index
    # category_to_index = {category: idx for idx, category in enumerate(unique_categories)}
    
    # # Map the original array to indices
    # indices = np.array([category_to_index[category] for category in transformed_arr])
    
    # # Create one-hot encoded matrix using identity matrix
    # one_hot = np.eye(len(unique_categories))[indices]
    
    return transformed_arr

def _filter_ohe_keypresses(keypresses, classes=[1.0, 10.0]):
    mask = np.concatenate(([True], keypresses[1:] != keypresses[:-1]))
    
    # Apply the mask to filter the array
    filtered_keypresses = keypresses[mask]
    
    mask = filtered_keypresses != 0.0
    
    filtered_keypresses = filtered_keypresses[mask]
    
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    
    # Initialize one-hot matrix with zeros
    one_hot = np.zeros((filtered_keypresses.size, len(classes)), dtype=int)
    
    # Map array elements to their class indices
    indices = np.vectorize(class_to_index.get)(filtered_keypresses)
    
    # Assign 1 to the appropriate positions
    one_hot[np.arange(filtered_keypresses.size), indices] = 1
    
    # differences = np.diff(filtered_keypresses)
    
    # # Convert differences to binary indicators (1 if different, 0 if same)
    # indicators = np.where(differences != 0, 1, 0)
    
    # # Prepend a 0 for the first element
    # indicators = np.insert(indicators, 0, 0)
    
    # one_hot = np.eye(2)[indicators]
    
    return one_hot


def get_key_sequences(data, N, window):
    df_X, df_y, path = data
    
    # Getting rid of slider tick (type 7) rows as it's only useful for positional embedding
    df_X = df_X[df_X['type_7.0'] != 1]
    
    # Getting rid of repeating spinner ticks (type 13) for the same reason
    df_X = df_X[df_X['type_13.0'] != 1]
    
    # feats_X = df_X[['type_1.0', 'type_6.0', 'type_12.0']].values
    feats_X = df_X[['type_1.0', 'type_6.0', 'type_12.0']]
    
    times_X = df_X['time'].values
    end_times_X = df_X['end_time'].values
    len_X = len(df_X)
    
    start_idx_X = 0
    end_idx_X = 0
    input_sequences = []
    
    # targets_y = df_y[['key_0.0', 'key_1.0', 'key_10.0', 'key_11.0']].values
    if df_y is not None:
        targets_y = _remove_overlap_keypresses(df_y[['keycode']].values)
        times_y = df_y['time'].values
        len_y = len(df_y)
        start_idx_y = 0
        end_idx_y = 0
        target_sequences = []
    
    
    for t in times_X:
        # Getting future objects within time window (t --- (t + window))
        while start_idx_X < len_X and times_X[start_idx_X] < t:
            start_idx_X += 1
        
        while end_idx_X < len_X and times_X[end_idx_X] <= t + window:
            end_idx_X += 1
        
        # Want the number of objects to filter out sections with no future objects i.e. empty section          
        num_future = end_idx_X - start_idx_X

        if num_future > 0:
            # If more than N, keep the N most recent objects
            if num_future > N:
                future_objs = feats_X[start_idx_X:start_idx_X + N]
                future_times = times_X[start_idx_X:start_idx_X + N]
                num_future = N
            else:
                future_objs = feats_X[start_idx_X:end_idx_X]
                future_times = times_X[start_idx_X:end_idx_X]
            
            min_delta_time = future_times.min() - t
            if min_delta_time == 0:
                # Getting normalized time offsets
                future_time_diff = (future_times - t) / window
                future_vector = np.hstack((future_objs, future_time_diff.reshape(-1, 1))).astype(np.float32)
                
                input_sequences.append(torch.tensor(future_vector, dtype=torch.float32))
                
                # Getting corresponding targets inside time window
                if df_y is not None:
                        
                    while start_idx_y < len_y and times_y[start_idx_y] < t - (1000/60):
                        start_idx_y += 1

                    while end_idx_y < len_y and times_y[end_idx_y] <= future_times.max() + (1000/60) * 2:
                        end_idx_y += 1
                    
                    if start_idx_y == end_idx_y:
                        start_idx_y -= 1
                        
                    keypresses = targets_y[start_idx_y:end_idx_y].astype(np.float32)
                    filtered_ohe_keypresses = _filter_ohe_keypresses(keypresses)
                    num_keypresses = len(filtered_ohe_keypresses)
                    
                    if num_keypresses < num_future:
                        # Usually cause by hitcircle transition to spinner, which does not require additional input
                        if future_objs.iloc[-1, 2] != 1:
                            print(f'Missing - objects: {num_future} keypresses: {num_keypresses}')
                    elif num_keypresses > num_future:
                        
                        # usually caused by a delayed input. So remove the first input which was part of previous object
                        filtered_ohe_keypresses = filtered_ohe_keypresses[1: 1 + num_future]
                        if len(filtered_ohe_keypresses) != num_future:
                            print(f'Extra - objects: {num_future} keypresses: {len(filtered_ohe_keypresses)}')
                    
                    
                    # if (len(filtered_ohe_keypresses) != len(future_objs)):
                    #     print(f'Mismatch - objects: {len(future_objs)} keypresses: {len(filtered_ohe_keypresses)}')
                    #     pass
                    
                    # filtered_keypresses = _filter_keypresses(keypresses, len(future_objs))
                    # Identify the first occurrence of each consecutive keypress
                    # Basically turns keypress holds into a single keypress
                    # row_changes = (key_target_vector[1:] != key_target_vector[:-1]).any(axis=1)

                    # # # Prepend True to include the first row and remove key0 (no key input)
                    # mask = np.concatenate(([True], row_changes))
                    # filtered = key_target_vector[mask]
                    # filtered = filtered[filtered != 0.0]
                    
                    # filtered_key_target_vector = removed_0
                    
                    
                    target_sequences.append(torch.tensor(filtered_ohe_keypresses, dtype=torch.float32))
        
    if df_y is not None:
        torch.save(input_sequences, f'{path}/key_input_seq.pt')
        torch.save(target_sequences, f'{path}/key_target_seq.pt')
    else:
        return input_sequences, times_X, end_times_X

def get_pos_sequences(data, N, window):
    df_X, df_y, path = data

    times_X = df_X['time'].values
    feats_X = df_X[['x_norm', 'y_norm', 'type_1.0', 'type_6.0', 
                              'type_7.0', 'type_12.0', 'type_13.0']].values


    # targets not defined when generating on unseen data
    # On unseen data time step (frame time) will be ~16.6 ms
    if df_y is not None:
        times_y = df_y['time'].values
        targets_y = df_y[['x_norm', 'y_norm']].values
    
        start_idx_y = 0
        end_idx_y = 0
        len_y = len(df_y)
    
        time_steps = df_y['time'].values
    else:
        end_time = times_X.max()
        time_steps = np.arange(0, end_time, (1000 / 60))
        time_steps = np.round(time_steps).astype(int)
    
    input_sequences = []
    target_sequences = []
    target_objects = []
    active_time_steps = []
    
    start_idx_X = 0
    end_idx_X = 0

    len_X = len(df_X)
    
    type_weight = np.array([10, 10, 1])

    for t in time_steps[0:]:
    
        # Getting future objects within time window (t --- (t + window))
        while start_idx_X < len_X and times_X[start_idx_X] < t:
            start_idx_X += 1
    
        while end_idx_X < len_X and times_X[end_idx_X] <= t + window:
            end_idx_X += 1
    
        num_active = end_idx_X - start_idx_X
        # Getting the previous object if its within some threshold of current time
        # I think this prevents the model from sometimes moving away from the nearest object too soon
        prev_obj = None
        if start_idx_X > 0:
            prev_time = times_X[start_idx_X - 1]
            if (t - prev_time) < 32:
                # num_active += 1
                prev_obj = feats_X[start_idx_X - 1]
                
        if num_active > 0:
            # If more than N, keep the N most recent objects
            if num_active > N:
                active_objs = feats_X[start_idx_X:start_idx_X + N]
                active_times = times_X[start_idx_X:start_idx_X + N]
                num_active = N
            else:
                active_objs = feats_X[start_idx_X:end_idx_X]
                active_times = times_X[start_idx_X:end_idx_X]
            
            # The model performs worse when there are few objects in a sequence
            # This will take the first object and pad the sequence to length N
            # Unsure if this is needed during training, but it improves inference performance
            if num_active < N:
                num_to_pad = N - num_active
                obj_padding = np.tile(active_objs[0], (num_to_pad, 1))
                active_objs = np.concatenate([active_objs[:1], obj_padding, active_objs[1:]], axis=0)
                time_padding = np.full(num_to_pad, active_times[0])
                active_times = np.concatenate([active_times[:1], time_padding, active_times[1:]], axis=0)
                
            if prev_obj is not None:
                active_objs = np.concatenate([[prev_obj], active_objs])
                active_times = np.concatenate([[prev_time], active_times])
                
            active_time_diff = (active_times - t) / window
            x_diff = np.diff(active_objs[:, 0], append=active_objs[-1, 0])
            y_diff = np.diff(active_objs[:, 1], append=active_objs[-1, 1])
            
            active_vector = np.hstack((
                active_objs, 
                active_time_diff.reshape(-1, 1), 
                x_diff.reshape(-1, 1), 
                y_diff.reshape(-1, 1)
                )).astype(np.float32)

            # Get targets for sequence on training data
            # Player targets is the human cursor position data
            # Object targets is the immediate object at time t (perhaps useful for hybrid loss)
            if df_y is not None:
                while start_idx_y < len_y and times_y[start_idx_y] < t:
                    start_idx_y += 1
                
                player_target_vector = targets_y[start_idx_y:start_idx_y + 2].astype(np.float32)

                target_sequences.append(torch.tensor(np.round(player_target_vector, decimals=4), dtype=torch.float32))
                
                # The immediate object is either the previous object due to 60hz sampling or the first future object
                immediate_idx = 0
                # immediate_times = active_times - t
                # if len(immediate_times) > 1:
                #     if np.abs(immediate_times[0]) > immediate_times[1]:
                #         immediate_idx = 1
                    
                immediate_object = active_vector[immediate_idx, [0, 1, 7]]
                
                # For hybrid loss, only hit circles and slider starts and slider ticks should be considered
                # Column 2 is hit circle, 3 is slider start and 4 is slider tick
                # More weight is applied on hit circles and slider starts
                valid_types = active_vector[immediate_idx, [2, 3, 4]]
                weight = np.sum(type_weight * valid_types)
                immediate_object = np.hstack((immediate_object, weight))
                
                target_objects.append(torch.tensor(np.round(immediate_object, decimals=4), dtype=torch.float32))
                
                # Embedding distance to hitobject relative to previous cursor position
                prev_pos = targets_y[start_idx_y]
                
                dist_x = (active_vector[:, 0] - prev_pos[0])
                dist_y = (active_vector[:, 1] - prev_pos[1])
                
                active_vector = np.hstack((active_vector, dist_x.reshape(-1, 1), dist_y.reshape(-1, 1)))
            else:
                active_time_steps.append(t)

            input_sequences.append(torch.tensor(np.round(active_vector, decimals=4), dtype=torch.float32))

    if df_y is not None:
        torch.save(input_sequences, f'{path}/pos_input_seq.pt')
        torch.save(target_sequences, f'{path}/pos_target_seq.pt')
        torch.save(target_objects, f'{path}/pos_target_obj.pt')
    else:
        return input_sequences, active_time_steps
    
def get_sequences(data, N, window, type):
    if type == "pos":
        get_pos_sequences(data, N, window)
    elif type == "key":
        get_key_sequences(data, N, window)
    
    
def read_sequences(path, type):
    input_seq = torch.load(f'{path}/{type}_input_seq.pt')
    target_seq = torch.load(f'{path}/{type}_target_seq.pt')
    
    if type == 'pos':
        target_obj = torch.load(f'{path}/pos_target_obj.pt')
        return [input_seq, target_seq, target_obj]
    
    return [input_seq, target_seq]

def get_set(set_path, type, regenerate=False):
    all_folders = [os.path.join(set_path, folder) for folder in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, folder))]
    if regenerate:
        set_folders = all_folders
    else:
        set_folders = [folder for folder in all_folders 
                            if not (os.path.exists(os.path.join(folder, f'{type}_input_seq.pt')) and 
                                    os.path.exists(os.path.join(folder, f'{type}_target_seq.pt')))]
    set = []
    
    with ProcessPoolExecutor() as executor:
            futures = []
            for folder in set_folders:
                futures.append(executor.submit(_get_Xy, folder, type))
            
            for future in tqdm(as_completed(futures), total = len(futures)):
                set.append(future.result())
    
    return set

# Writes to disk to avoid data transfer between processes
def create_set_sequences(set, N, react_time, type):
    with ProcessPoolExecutor() as executor:
        futures = []
        for data in set:
            futures.append(executor.submit(get_sequences, data, N, react_time, type))
        
        for future in tqdm(as_completed(futures), total = len(futures)):
            pass

def get_set_sequences(set_path, type):
    set_folders = [os.path.join(set_path, folder) for folder in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, folder))]
    input_sequences = []
    target_sequences = []
    target_objects = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for folder in set_folders:
            futures.append(executor.submit(read_sequences, folder, type))
        
        for future in tqdm(as_completed(futures), total = len(futures)):
            sequences = future.result()
            input_sequences.append(sequences[0])
            target_sequences.append(sequences[1])
            target_objects.append(sequences[2])
            
    return input_sequences, target_sequences, target_objects