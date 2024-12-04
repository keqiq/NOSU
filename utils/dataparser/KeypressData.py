from .DataParser import DataParser
from osrparse import Replay
import numpy as np
import torch

# KeypressData transforms .osu map files and .osr replay files into tensors used by keypress model
class KeypressData(DataParser):
    def __init__(self, set_paths, c_size, t_size):
        super().__init__(set_paths, c_size, t_size)
        self.subclass = 'key'
        
    # Converts .osr replay files into numpy 2d array
    @staticmethod
    def parse_replay(replay_path):
        replay = Replay.from_path(replay_path)
        replay_data = replay.replay_data
        results = []
        time = replay_data[1].time_delta

        for event in replay_data[2:]:
            time += event.time_delta
            results.append([event.x, event.y, time, event.keys.value])
        
        return np.array(results), False
    
    # When both key1 (1) and key2 (10) are pressed down a new keycode (11) is used to represent the state
    # For simplicity, this function will replace said keycode with key1 or key2
    @staticmethod
    def _remove_overlap_keypresses(keypresses):
        keypresses = keypresses.flatten()
        
        # To determine whether to replace with key1 or key2 the previous key needs to be determined
        prev_key = None
        
        for i in range(len(keypresses)):
            curr_key = keypresses[i]
            
            if curr_key == 11.0:
                # Overlap key 11 resulting from holding down key1 then pressing key2
                if prev_key == 1.0:
                    # Replace with key2
                    keypresses[i] = 10.0
                
                # Overlap key 11 resulting from holding down key2 then pressing key1
                elif prev_key == 10.0:
                    # Replace with key1
                    keypresses[i] = 1.0
            
            else:
                # Update prev_key if curr_key is not 11.0 and not 0.0
                if curr_key in [1.0, 10.0]:
                    prev_key = curr_key
                    
        return keypresses
    
    # This function will filter and array of keypresses and keep the non consequtive keypresses then one-hot encode it
    # For example a sequence containing [key1, key1, key0, key0, key0, key1, key2, key2] will result in [key1, key1, key2]
    # The size of the sequence should match the number of active objects
    # This simplifies the learning significantly as the model need only learn which keys to press down for each object
    @staticmethod
    def _filter_ohe_keys(keys, classes=[1.0, 10.0]):
        # mask the positions where the keypresses change
        mask_change = np.concatenate(([True], keys[1:] != keys[:-1]))
        masked_keys = keys[mask_change]
        
        # Filter out key0 (no input) and keep only key1 and key2
        mask_input = masked_keys != 0.0
        masked_keys = masked_keys[mask_input]
        
        # No input
        if len(masked_keys) == 0:
            return []
        
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    
        # Initialize one-hot matrix with zeros
        one_hot = np.zeros((masked_keys.size, len(classes)), dtype=int)
        
        # Map array elements to their class indices
        indices = np.vectorize(class_to_index.get)(masked_keys)
        
        # Assign 1 to the appropriate positions
        one_hot[np.arange(masked_keys.size), indices] = 1
        
        return one_hot
         
    # Function to generate sequences given a set containing X, y(optional) and path
    def generate_sequences(self, set):
        df_X, df_y, path = set
        
        # Removing slider tick (type 7) rows as it's only useful for positional embedding
        df_X = df_X[df_X['type_7.0'] != 1]
        
        # Removing spinner ticks (type 13) for the same reason
        df_X = df_X[df_X['type_13.0'] != 1]
        
        # Extracting relevant columns from df_X for keypress sequence inputs
        feats_X = df_X[['type_1.0', 'type_6.0', 'type_12.0']]
        times_X = df_X['time'].values
        end_times_X = df_X['end_time'].values
        
        num_objects = len(df_X)
        
        # Index counters used to select objects from X during loop
        start_idx_X = 0
        end_idx_X = 0
        
        # Initializing containers for output
        input_sequences = []
        target_sequences = []
        
        if df_y is not None:
            # Extracting relevant columns from df_y for keypress sequence targets
            targets_y = self._remove_overlap_keypresses(df_y[['keycode']].values)
            times_y = df_y['time'].values
            num_targets = len(df_y)
            
            # Index counters used to select targets from y during loop
            start_idx_y = 0
            end_idx_y = 0
            
        # Loop to iterate over every time in times_X t
        # The input sequence contains up to c_size objects from X within t and t + t_size
        # The target sequence (training) contains all non consequtive keypresses from y within a window
        # The window is between t and time value of the last object in the current selection
        for t in times_X:
            
            # Setting pointers to contain objects within time window (t --- t + t_size)
            while start_idx_X < num_objects and times_X[start_idx_X] < t:
                start_idx_X += 1
            
            while end_idx_X < num_objects and times_X[end_idx_X] <= t + self.t_size:
                end_idx_X += 1
                
            num_active = end_idx_X - start_idx_X
            
            if num_active > 0:
                
                # Extract objects within the time window into active_objs and likewise for time
                # If there are more than c_size objects in the window, keep only the first c_size objects
                if num_active > self.c_size:
                    active_objs = feats_X[start_idx_X : start_idx_X + self.c_size]
                    active_times = times_X[start_idx_X : start_idx_X + self.c_size]
                    num_active = self.c_size
                else:
                    active_objs = feats_X[start_idx_X : end_idx_X]
                    active_times = times_X[start_idx_X : end_idx_X]
                
                # Calculating the time delta between t and object time then normalizing it
                active_time_delta = (active_times - t) / self.t_size
                
                # Stacking all object features
                active_vector = np.hstack((active_objs, active_time_delta.reshape(-1, 1))).astype(np.float32)
                
                # Getting target sequences as part of training data
                if df_y is not None:
                    
                    # Setting pointers to contain targets within time window ((t - a) --- (t_last + b))
                    # constant a and b are used to widen the time window as player inputs can be early or delayed
                    while start_idx_y < num_targets and times_y[start_idx_y] < t - (1000/60): # a = 16.6 or 1 frame
                        start_idx_y += 1
                    
                    while end_idx_X < num_targets and times_y[end_idx_y] <= active_times.max() + (1000/60)*2: # b = 33.3 or 2 frames
                        end_idx_y += 1
                    
                    # Unsure if this is needed
                    if start_idx_y == end_idx_y:
                        start_idx_y -= 1
                    
                    keys = targets_y[start_idx_y : end_idx_y].astype(np.float32)
                    filtered_ohe_keys= self._filter_ohe_keys(keys)
                    num_keys = len(filtered_ohe_keys)
                    
                    # _filter_ohe_keys should return a sequence with the same size as active_obj in most cases
                    # Due to game mechanics and human delays there may be exceptions
                    # These conditionals will check if exceptions occur
                    
                    # If for some reason there is no input, skip current t as it is invalid
                    if num_keys == 0:
                        continue
                    if num_keys < num_active:
                        # Usually caused by hitcircle transition to spinner, which does not require additional input
                        if active_objs.iloc[-1, 2] != 1:
                            print(f'Missing - objects: {num_active} keypresses: {num_keys}')
                    
                    elif num_keys > num_active:
                        # Usually caused by a delayed input
                        # In this case remove the first input which was part of previous object
                        filtered_ohe_keys = filtered_ohe_keys[1: 1 + num_active]
                        if len(filtered_ohe_keys) != num_active:
                            print(f'Extra - objects: {num_active} keypresses: {len(filtered_ohe_keys)}')
                            
                    # Convert the target vector into tensor and append sequence to container
                    target_sequences.append(torch.tensor(filtered_ohe_keys, dtype=torch.float32))
            
                # Convert the input vector into tensor and append sequence to container
                input_sequences.append(torch.tensor(active_vector, dtype=torch.float32))
                
        # When creating training data, ProcessPoolExecutor is used for generating sequences in parallel
        # Pickling the return data is avoided by saving the data to disk then fetching it     
        if df_y is not None:
            torch.save(input_sequences, f'{path}/key_input_seq.pt')
            torch.save(target_sequences, f'{path}/key_target_seq.pt')
        else:
            return [input_sequences, times_X, end_times_X] 
        
    # Function to retreive data files from generate_sequence
    @staticmethod
    def read_sequences(path):
        input_seq = torch.load(f'{path}/key_input_seq.pt')
        target_seq = torch.load(f'{path}/key_target_seq.pt')
        return [input_seq, target_seq, []]
    
    # # Function to format results from read_sequences
    # @staticmethod
    # def format_results(results):
    #     train = results['train']
    #     # Combine train datasets into single dataset
    #     combined_input, combined_target = [sum(x, []) for x in zip(*train)]
        
    #     formatted_results = {
    #         'train': [combined_input, combined_target],
    #         'valid': results['valid']
    #     }
        
    #     return formatted_results