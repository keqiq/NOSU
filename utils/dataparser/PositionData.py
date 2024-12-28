from .DataParser import DataParser
from osrparse import Replay
import numpy as np
import torch

"""
PositionData transforms .osu map files and .osr replay files into tensors used by position model
"""
class PositionData(DataParser):
    
    def __init__(self, set_paths, config, regen=False):
        super().__init__(set_paths, config['pos_context_size'], config['pos_time_window'], regen)
        self.subclass = 'pos'
    
    """
    Converts .osr replay files into numpy 2d array
    Checks for Hard Rock (hr) mod, which flips hitobjects on the x-axis
    """
    @staticmethod
    def parse_replay(replay_path):
        replay = Replay.from_path(replay_path)
        replay_data = replay.replay_data
        
        # Check for hard rock, and pass this flag onto future steps
        hr = False
        mods_flag = replay.mods.value
        if ((mods_flag >> 4) & 1):
            hr = True
        results = []
        time = replay_data[1].time_delta
        
        i = 2
        while i < len(replay_data):
            time_delta = replay_data[i].time_delta
            
            """
            Some samples intervals are irregular < 16ms due to keypress inputs which will impact training
            This will merge those samples so that most intervals are 16 or 17 ms apart
            """
            if time_delta < 16:
                look_foward = 1
                total_time_delta = time_delta
                last_event = replay_data[i]
                while total_time_delta < 16 and (i + look_foward) < len(replay_data):
                    total_time_delta += replay_data[i + look_foward].time_delta
                    last_event = replay_data[i + look_foward]
                    look_foward += 1
                    
                time += total_time_delta
                event = last_event
                i += look_foward

            else:
                event = replay_data[i]
                time += time_delta
                i +=1
                
            results.append([event.x, event.y, time, event.keys.value])
        
        return np.array(results), hr
    
    """
    Function to generate sequences given a set containing X, y(optional) and path
    """
    def generate_sequences(self, set):
        df_X, df_y, path = set
        
        # Extracting relevant columns from df_X for position sequence inputs
        times_X = df_X['time'].values
        feats_X = df_X[['x_norm', 'y_norm', 'type_1.0', 'type_6.0', 
                              'type_7.0', 'type_12.0', 'type_13.0', 'buzz']].values
        
        """
        Extracting relevant columns from df_y for position sequence targets
        time_steps contains the time at which there is a sample for cursor movement 
        Targets (df_y) is not available during inference
        During inference create dummy time_steps at every ~16.6ms
        """
        if df_y is not None:
            times_y = df_y['time'].values
            targets_y = df_y[['x_norm', 'y_norm']].values
            time_steps = df_y['time'].values
        else:
            end_time = times_X.max()
            time_steps = np.arange(0, end_time, (1000 / 60))
            time_steps = np.round(time_steps).astype(int)
            
        num_objects = len(times_X)
        num_targets = len(time_steps)
        
        # Initializing containers for output
        input_sequences = []
        target_sequences = []
        target_objects = []
        active_time_steps = []
        
        # Index counters used to select objects from X during loop
        start_idx = 0
        end_idx = 0
        
        """        
        Loop to iterate over every time_step t
        The input sequence contains c_size objects from X within t and t + t_size
        The target sequence (training) contains the cursor sample from y at t and the previous sample
        The target object is the object with the smallest time delta to t
        t is added to active_time_steps (inference) if the input sequence is not empty
        In other words, t is not active if there are no objects within t and t + t_size
        """
        
        for y_idx, t in enumerate(time_steps):
            
            # Setting pointers to contain objects within time window (t --- t + t_size)
            while start_idx < num_objects and times_X[start_idx] < t:
                start_idx += 1
                
            while end_idx < num_objects and times_X[end_idx] <= t + self.t_size:
                end_idx += 1
                
            num_active = end_idx - start_idx
            
            # Due to 60hz sampling, the nearest object could be outside of the current time window
            # This will keep the previous object before the window within some time threshold
            prev_obj = None
            if start_idx > 0:
                prev_time = times_X[start_idx - 1]
                if (t - prev_time) < 16:
                    prev_obj = feats_X[start_idx - 1]
                    
            if num_active > 0:
                
                # Extract objects within the time window into active_objs and likewise for time
                # If there are more than c_size objects in the window, keep only the first c_size objects
                if num_active > self.c_size:
                    active_objs = feats_X[start_idx : start_idx + self.c_size]
                    active_times = times_X[start_idx : start_idx + self.c_size]
                    num_active = self.c_size
                else:
                    active_objs = feats_X[start_idx : end_idx]
                    active_times = times_X[start_idx : end_idx]
                    
                # Adding the previous object into the input sequence
                if prev_obj is not None:
                    active_objs = np.concatenate([[prev_obj], active_objs])
                    active_times = np.concatenate([[prev_time], active_times])
                    
                """
                The model performs worse when there are few objects in the input sequence
                Could be due to most maps having relatively few portions where only few objects are present
                and thus small sequences are dilute in the training examples
                This check will take the first object and pad the sequence to c_size + a
                
                Also, duplicate the first object proportional to self.csize (a) to emphasize importance
                First object is generally a good indicator for good predictions
                """
                if num_active < self.c_size + int(self.c_size/2):
                    num_to_pad = (self.c_size + int(self.c_size/2)) - num_active
                    obj_padding = np.tile(active_objs[0], (num_to_pad, 1))
                    active_objs = np.concatenate([active_objs[:1], obj_padding, active_objs[1:]], axis=0)
                    time_padding = np.full(num_to_pad, active_times[0])
                    active_times = np.concatenate([active_times[:1], time_padding, active_times[1:]], axis=0)

                # Calculating the time delta between t and object time then normalizing it
                active_time_delta = (active_times - t) / self.t_size
                
                # Calculating the distance between objects
                diff_x = np.diff(active_objs[:, 0], append=active_objs[-1, 0])
                diff_y = np.diff(active_objs[:, 1], append=active_objs[-1, 1])
                
                # Stacking all object features
                active_vector = np.hstack((
                    active_objs, 
                    active_time_delta.reshape(-1, 1), 
                    diff_x.reshape(-1, 1), 
                    diff_y.reshape(-1, 1)
                    )).astype(np.float32)
                
                # Getting target sequences as part of training data
                # Or appending t to active_time_steps for inference
                if df_y is not None and y_idx < num_targets:
                    
                    # Player targets is the human cursor position data from replay
                    player_target_vector = targets_y[y_idx - 1 : y_idx + 1]
                    if (len(player_target_vector) < 2):
                        continue
                    target_sequences.append(
                        torch.tensor(player_target_vector,
                        dtype=torch.float32
                    ))
                    
                    # Object targets is the immediate (smallest time value) object at time t used in hybrid loss
                    immediate_idx = 0
                    immediate_vector = active_vector[immediate_idx]
                    
                    """
                    For hybrid loss, only hit circles and slider starts and slider ticks should be considered
                    Column 2 is hit circle, 3 is slider start and 4 is slider tick
                    """
                    
                    target_objects.append(
                        torch.tensor(immediate_vector,
                        dtype=torch.float32        
                    ))
                    
                    # Calculating distance to object relative to previous cursor position sample
                    prev_pos = targets_y[y_idx - 1]
                    dist_x = (active_vector[:, 0] - prev_pos[0])
                    dist_y = (active_vector[:, 1] - prev_pos[1])
                    
                    # Stacking distances with active vector
                    active_vector = np.hstack((active_vector, dist_x.reshape(-1, 1), dist_y.reshape(-1, 1)))
                    
                else:
                    active_time_steps.append(t)
                
                input_sequences.append(
                    torch.tensor(active_vector,
                    dtype=torch.float32
                ))
        
        # When creating training data, ProcessPoolExecutor is used for generating sequences in parallel
        # Pickling the return data is avoided by saving the data to disk then fetching it
        if df_y is not None: 
            torch.save(input_sequences, f'{path}/pos_input_seq.pt')
            torch.save(target_sequences, f'{path}/pos_target_seq.pt')
            torch.save(target_objects, f'{path}/pos_target_obj.pt')
        else:
            return input_sequences, active_time_steps, None