from ..map.HitObjects import HitObjects
from ..map.MapParams import MapParams
from ..map.TimingPoints import TimingPoints

import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence

"""
DataParser turns .osu files and replay files into tensors used during training and inference
This is the super class for PositionData and KeypressData providing some type agnoistic functions
The structure of tensors for the position model and keypress model is different
"""
class DataParser:
    def __init__(self, set_paths, config, regen=False):
        self.set_paths = set_paths
        self.c_size = config[f'{self.subclass}_context_size']
        self.t_size = config[f'{self.subclass}_time_window']
        self.buzz_thresholds = {
            'L' : config['linear_buzz_threshold'],
            'P' : config['circle_buzz_threshold'],
            'B' : config['bezier_buzz_threshold']
        }
        self.regen = regen
        self.W = 512
        self.H = 384
    
    """
    Function to convert .osu map files into numpy 2d array
    """
    def _parse_map(self, map_path, hr):
        with open(map_path, 'r', encoding='utf-8') as file:
            lines = file.read()
        
        sections = lines.split('\n\n')

        for section in sections:
            content = section.split('\n')
            header = content.pop(0)
            if header == '': header = content.pop(0)
            
            # Need the stack leniency value for stacker
            # Since general section is the first section we can just pass this value to MapParams
            if header == '[General]':
                stack_leniency = content[5]
            
            # Difficulty values
            if header == '[Difficulty]':
                mp = MapParams(content, stack_leniency)

            # Timing points
            elif header == '[TimingPoints]':
                tp = TimingPoints(content)

            # Hitobject parsing
            elif header == '[HitObjects]':
                if content[-1] == '': content.pop(-1)
                ho = HitObjects(mp, tp, content, hr, self.buzz_thresholds)
        
        return ho.get_data()

    """ 
    Converts the numpy array from _parse_map into a dataframe
    Also perfroms some preprocessing on features
    Useful for analysing the data and required for sequence generation
    Either provide a folder containing the map (training) or the map path (inference)
    """
    def get_X(self, folder_path, hr, map_path=None):
        if map_path:
            _, feats = self._parse_map(map_path, hr)
        else:
            _, feats = self._parse_map(list(Path(folder_path).glob('*.osu'))[0], hr)
        
        x = pd.DataFrame(feats, columns=[
            'x', 'y', 'time', 'type', 'end_time', 'buzz'
        ])
        
        x['delta_time'] = (x['time'].diff()).fillna(0)
        
        # Normalizing x,y according to osu playfield dimension (512 x 384)
        x['x_norm'] = x['x'] / self.W
        x['y_norm'] = x['y'] / self.H
        
        # One-Hot encode type categorical feature
        types = ['1.0', '6.0', '7.0', '12.0', '13.0']
        type_ohe = pd.get_dummies(x['type'], prefix='type').reindex(columns=[f'type_{t}' for t in types], fill_value=0)
        x = pd.concat([x, type_ohe], axis=1)
        x = x.drop(['x', 'y', 'type'], axis=1)
        
        return x
    
    """
    Converts the numpy array from parse_replay into a dataframe
    Similar to get_X but is only needed for training
    """
    def get_y(self, path):
        target, hr = self.parse_replay(list(Path(path).glob('*.osr'))[0])

        y = pd.DataFrame(target, columns=[
        'x', 'y', 'time', 'keycode'
        ])
        y['delta_time'] = (y['time'].diff()).fillna(0)
        y['x_norm'] = y['x'] / self.W
        y['y_norm'] = y['y'] / self.H
        
        """
        Compressing keycodes where different keycode refer to same action
        Bitwise combination of keys/mouse buttons pressed 
        (M1 = 1, M2 = 2, K1 = 4, K2 = 8, Smoke = 16) 
        (K1 is always used with M1; K2 is always used with M2: 1+4=5; 2+8=10)
        From osu wiki
        """
        y['keycode'] = y['keycode'].replace({
            5: 1,   #key 1
            2: 10,  #key 2
            15: 11, #key 1 + 2
            16: 0,  #smoke
            21: 1,  #key1 + smoke
            26: 10, #key2 + smoke
            31: 11  #key1 + key2 + smoke
        })
        
        return y, hr
    
    """
    Function to get y first then X during training
    Note in order to get X we first need the hr flag from the replay file
    """
    def _get_Xy(self, path):
        df_y, hr = self.get_y(path)
        df_X = self.get_X(path, hr)
        return [df_X, df_y, path]
    
    """
    Function to read .pth tensor files from disk
    """
    def _read_sequences(self, path):
        input_seq = torch.load(f'{path}/{self.subclass}_input_seq.pt')
        target_seq = torch.load(f'{path}/{self.subclass}_target_seq.pt')
        if self.subclass == 'pos':
            target_obj = torch.load(f'{path}/pos_target_obj.pt')
        else:
            target_obj = []
        return [input_seq, target_seq, target_obj]
    
    
    """
    Function to read .osu map files and .osr replay files in parallel
    Provide the path to the folder containing folders which contains the map and associated replay
    """
    def _get_dataframes(self, set_path):
        all_folders = [os.path.join(set_path, folder) for folder in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, folder))]
        
        # With regen flag to true, get all dataframes
        # Else, only get dataframes from folders which do not have data present
        if self.regen:
            set_folders = all_folders
        else:
            set_folders = [folder for folder in all_folders 
                            if not (os.path.exists(os.path.join(folder, f'{self.subclass}_input_seq.pt')) and 
                                    os.path.exists(os.path.join(folder, f'{self.subclass}_target_seq.pt')))]
            
        set = []
        
        failures = 0
        with ProcessPoolExecutor() as executor:
            futures = []
            for folder in set_folders:
                futures.append(executor.submit(self._get_Xy, folder))
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                # Only append successful results
                try:
                    result = future.result()
                    set.append(result) 
                except Exception as e:
                    failures += 1 
                    print(f"Error occurred for future: {e}")

        print(f"{failures} failures out of {len(set_folders)} total tasks.")
        
        return set
    
    """
    Function to generate sequences given a set containing X, y and path in parallel
    There is no return as the resulting data is written to disk to avoid pickling between processes
    """
    def _generate_set_sequences(self, sets):
        with ProcessPoolExecutor() as executor:
            failures = 0
            futures = []
            for set in sets:
                futures.append(executor.submit(self.generate_sequences, set))
                
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception as e:
                    failures += 1
                    print(f"Error occurred for a task: {e}")

        print(f"{failures} failures out of {len(futures)} total tasks.")
    
    """
    Function to read data files from disk from generate_set_sequences in parallel
    """
    def _get_set_sequences(self, set_path):
        set_folders = [os.path.join(set_path, folder) for folder in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, folder))]
        results = []
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for folder in set_folders:
                futures.append(executor.submit(self._read_sequences, folder))
                
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.append(future.result())
                 
        return results
    
    """
    Wrapper function to handle every conversion step
    """
    def generate(self):
        datasets = {
            'train': None,
            'valid': None
        }
        # Step 1 - Converting to dataframes
        for set in self.set_paths:
            print(f'Converting {set} data')
            data = self._get_dataframes(self.set_paths[set])
            datasets[set] = data
        
        # Step 2 - Creating list of sequence tensors
        for set in datasets:
            if len(datasets[set]) > 0:
                print(f'Generating {set} sequences')
                self._generate_set_sequences(datasets[set])
                
        # Step 3 - Fetching results from previous step and structure return accordingly
        results = {
            'train': None,
            'valid': None
        }
        for set in self.set_paths:
            print(f'Retrieving {set} sequences')
            results[set] = self._get_set_sequences(self.set_paths[set])
        
        # Step 4 - Format results to be passed into Dataloaders
        formatted_results = self.format_results(results)
        
        return formatted_results
    
    """
    Function to generate sequences for one map (inference)
    """
    def generate_one(self, path):
        df_X = self.get_X(None, False, path)
        
        sequences, times, end_times = self.generate_sequences([df_X, None, path])
        
        inputs = []
        for seq in sequences:
            input_length = torch.tensor([seq.size(0)], dtype=torch.long)
            padded_input = pad_sequence([seq], batch_first=True, padding_value=0)
            inputs.append([padded_input, input_length])
            
        return inputs, times, end_times
        
    """
    Function to format results from read_sequences to be passed into dataloader
    """
    @staticmethod
    def format_results(results):
        train = results['train']
        # Combine train datasets into single dataset
        combined_input, combined_target, combined_object = [sum(x, []) for x in zip(*train)]
        
        formatted_results = {
            'train': [combined_input, combined_target, combined_object],
            'valid': results['valid']
        }
        
        return formatted_results
    