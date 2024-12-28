import customtkinter as ctk
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import contextlib
import threading
from Model.modelmanager.PositionModel import PositionModel
from Model.modelmanager.KeypressModel import KeypressModel
from utils.dataparser.PositionData import PositionData
from utils.dataparser.KeypressData import KeypressData
from utils.dataparser.OSUDataloader import OSUDataloader
from utils.PostProcess import post_process, save_replay
import torch

# Appearance mode and color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
selected_color = '#7F46FA'
unselected_color = '#3B3B3B'
CONFIG_PATH = 'config.json'
VERSION = 'Beta_1.0'

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(f'Neurosu! - {VERSION}')
        self.geometry("1600x1000")

        # Loading configuration file from directory root
        with open(CONFIG_PATH, 'r') as f:
            self.config = json.load(f)

        # Check if a song path is specified
        # If not default to osu!'s default install directory
        if self.config['song_path']:
            self.song_path = self.config['song_path']
        else:
            home_dir = os.path.expanduser("~")
            song_path = os.path.join(home_dir, "Appdata", "Local", "osu!", "Songs")
            self.song_path = song_path
        
        self.selected_map = None
        self.train_log = None
        self.button_generate_replay = None
        self.button_train_position = None
        self.button_train_keypress = None
        self.busy = False
        self.selected_color = selected_color
        self.unselected_color = unselected_color
        
        # Initializing tabs for inference and training
        self.tabview = ctk.CTkTabview(self, fg_color='transparent')
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Add tabs
        self.tab_inference = self.tabview.add("Inference")
        self.tab_training = self.tabview.add("Training")
        
        # Build tabs
        from gui.tab_inference import build_inference_tab
        from gui.tab_training import build_training_tab
        
        build_inference_tab(self.tab_inference, self)
        build_training_tab(self.tab_training, self)
        
        # Display device information 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if 'cuda' in str(self.device):
            device_name = f"[CUDA] {torch.cuda.get_device_name(0)}"
            label_color = "green"
        else:
            device_name = "[CPU]"
            label_color = "orange"

        # Floating label in the top-right corner
        self.label_device = ctk.CTkLabel(
            self,
            text=f"Device: {device_name}",
            text_color=label_color
        )
        self.label_device.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=10)
    
    """
    Logging function which can be called from children to write to self.train_log
    """
    def write_to_log(self, box, text: str):
        # Make sure the log box is in normal state before inserting
        box.configure(state="normal")
        box.insert("end", text + '\n')
        box.see("end")  # auto-scroll to the bottom
        
        box.configure(state="disabled")
    
    """
    Training function takes the selected configuration and model type
    Will create a thread for data parsing and training to not freeze ui
    """
    def start_training(self, config, name, type):
        # Update busy state and disable buttons
        self.busy = True
        self.update_buttons()
        
        self.save_config('name', name)
        paths = self.load_config()['paths']
        selected_config = config.get()
        self.write_to_log(self.train_log, f'Training {type} with config {selected_config}')
        updated_config = self.parse_config(self.config['train']['default'], self.config['train'][selected_config])
        combined_config = {}
        for key, sub_dict in updated_config.items():
            combined_config.update(sub_dict)
        
        if type == 'position':
            data = PositionData(paths, updated_config['Data'], True)
            model = PositionModel(f'{self.config['name']}', combined_config, self.device)
        else :
            data = KeypressData(paths, updated_config['Data'], True)
            model = KeypressModel(f'{self.config['name']}', combined_config, self.device)

        dataloader = OSUDataloader(data, updated_config['Data'], True)
        
        thread = threading.Thread(
            target=self._train_thread,
            args=(paths, dataloader, model)
        )
        
        thread.start()
    
    """
    During training all print statements to stdout will be redirected to self.train_log
    Once the training finishes the model name, hyperparameters, weights will be save to config['save_path']
    """
    def _train_thread(self, paths, dataloader, model):
        with self.redirect_stdout_to_textbox(self.train_log):
            print("[PARSING DATA]")
            train_loader, valid_loader = dataloader.get_loaders(paths)
            print("[TRAINING]")
            model.train(train_loader, valid_loader)
            model.save_model(self.config['save_path'])
            # Update busy state and enable buttons
            self.busy = False
            self.update_buttons()
    
    """
    References kept for inference
    Function is called on startup to load model from config['models']
    Also called when choosing a different model
    """
    def load_model(self, model_type):
        model = torch.load(self.config['models'][model_type])
        if model_type == 'Position':
            self.pos_model = model
            self.pos_name = model['model_name']
            self.pos_hyperparams = model['hyperparameters']
            self.pos_weights = model['model_weights']
        elif model_type == 'Keypress':
            self.key_model = model
            self.key_name = model['model_name']
            self.key_hyperparams = model['hyperparameters']
            self.key_weights = model['model_weights']
    
    """
    Once a map is selected from song_select widget check if the map has mode 0 for standard
    If the map is standard then update reference to self.selected_map if not clear it
    """
    def process_map_file(self, map_path):
        with open(map_path, 'r', encoding='utf-8') as file:
            lines = file.read()
        
        general = lines.split('\n\n')[1]
        mode = general.split('\n')[7].split(':')[1]
        
        if mode == ' 0':
            self.selected_map = map_path
        else:
            self.selected_map = None

    """
    Inference function to generate replay from reference stored in self.selected_map
    Also creates a thread for predictions
    """
    def generate_replay(self):
        # Display progress bars
        self.frame_pb.pack(fill='x', padx=10, pady=10)
        # Update busy state and disable buttons
        self.busy = True
        self.update_buttons()
        
        pos_data = PositionData(None, {
            'pos_context_size': self.pos_hyperparams['context_size'],
            'pos_time_window': self.pos_hyperparams['time_window']
            })
        key_data = KeypressData(None, {
            'key_context_size': self.key_hyperparams['context_size'],
            'key_time_window': self.key_hyperparams['time_window']
        })
        pos_model = PositionModel(self.pos_name, None, self.device, self.pos_model)
        key_model = KeypressModel(self.key_name, None, self.device, self.key_model)
        
        thread = threading.Thread(
            target=self._inference_thread,
            args=(pos_data, pos_model, key_data, key_model)
        )
        
        thread.start()
    
    """
    During inference display a progress bar which is updated through self._update_progress
    When inference finishes save the replay .osr file to config['replay_path'] then reset and hide progress bars
    """
    def _inference_thread(self, pos_data, pos_model, key_data, key_model):
        
        def update_progress(progress, type):
            self._update_progress_bars(progress, type)
        
        pos_input, pos_time, _ = pos_data.generate_one(self.selected_map)
        key_input, key_time, key_end_time = key_data.generate_one(self.selected_map)
        
        pos_pred = pos_model.predict(pos_input, update_progress)
        key_pred = key_model.predict(key_input, update_progress)
        
        predictions = post_process(pos_pred, key_pred, pos_time, key_time, key_end_time)
        save_replay(predictions, 'replay_template.osr', self.selected_map, pos_model.name, key_model.name)
        
        # Update busy state and enable buttons
        self.busy = False
        self.update_buttons()
        # Reset and clear progress bars
        self._update_progress_bars(0.0, 'all')
        self.frame_pb.pack_forget()
    
    """
    Function to update progress bars
    """
    def _update_progress_bars(self, progress, type):
        if type == 'Position' or type == 'all':
            self.pb_pos.set(progress)
        elif type == 'Keypress' or type == 'all':
            self.pb_key.set(progress)
        self.update_idletasks()
    
    def save_config(self, key, config):
        self.config[key] = config
        with open(CONFIG_PATH, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    """
    User created configurations stores parameters only if it differs from default configuration
    This function will return a complete configuration based on default and custom
    """
    def parse_config(self, default, custom):
        merged = {}
        
        for key, value in default.items():
            if key in custom:
                if isinstance(value, dict) and isinstance(custom[key], dict):
                    merged[key] = self.parse_config(value, custom[key])
                else:
                    merged[key] = custom[key]
            else:
                merged[key] = value
        
        return merged
    
    """
    Function to toggle buttons based on self.busy state and if self.selected_map is chosen
    """
    def update_buttons(self):
        if self.selected_map and not self.busy and self.pos_name and self.key_name:
            self.button_generate_replay.configure(state='normal')
        else:
            self.button_generate_replay.configure(state='disabled')
        
        if not self.busy:
            self.button_train_position.configure(state='normal')
            self.button_train_keypress.configure(state='normal')
        else:
            self.button_train_position.configure(state='disabled')
            self.button_train_keypress.configure(state='disabled')
        
    @contextlib.contextmanager
    def redirect_stdout_to_textbox(self, textbox):
   
        class TextBoxWriter:
            def write(self, text):
                # Insert text into the textbox
                textbox.configure(state="normal")
                textbox.insert("end", text)
                textbox.see("end")
                textbox.configure(state="disabled")

        old_stdout = sys.stdout
        sys.stdout = TextBoxWriter()
        try:
            yield
        finally:
            # Restore the original stdout
            sys.stdout = old_stdout
            
if __name__ == "__main__":
    app = App()
    app.mainloop()