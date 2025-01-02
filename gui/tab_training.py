import customtkinter as ctk
from tkinter import filedialog
from gui.widget_configurations import build_config_widget
import os
global_app = None

"""
Function to select folder containing .osu beatmap files and .osr replay files
Updates app.config['paths'] with selected folder path
"""
def _browse_folder(set, entry):
    folder_selected = filedialog.askdirectory(title=f'Select {set} folder', initialdir=global_app.config['paths'][set])
    if folder_selected:
        entry.delete(0, "end")
        entry.insert(0, folder_selected)
        
    config = global_app.config['paths']
    config[set] = folder_selected
    global_app.save_config('paths', config)
    _get_folder_contents(set, folder_selected)
    
def _get_folder_contents(set, path):
    try:
        # Get all subfolders in the path
        subfolders = [
            os.path.join(path, folder)
            for folder in os.listdir(path)
            if os.path.isdir(os.path.join(path, folder))
        ]

        # Count subfolders containing .osu files
        num_files = 0
        for subfolder in subfolders:
            files = os.listdir(subfolder)
            has_osu = any(file.endswith('.osu') for file in files)
            has_osr = any(file.endswith('.osr') for file in files)
            
            if has_osu and has_osr:
                num_files += 1

        # Write to log
        global_app.write_to_log(global_app.train_log, f'{num_files} beatmap replay pairs in {set}')

    except Exception as e:
        global_app.write_to_log(global_app.train_log, f'Error checking "{set}": {e}')

"""
Training tab build function
Displays:
Train data path
Valid data path
Configuration widget
Train buttons
Log textbox
"""
def build_training_tab(parent, app):
    global global_app
    global_app = app
    content_frame = ctk.CTkFrame(parent)
    content_frame.pack(side="left", fill="both", padx=10, pady=10)
    
    # Train
    frame_train_path = ctk.CTkFrame(content_frame, fg_color='transparent')  
    frame_train_path.pack(fill="x", padx=10, pady=5)

    label_train_path = ctk.CTkLabel(frame_train_path, text="Train Folder:")
    label_train_path.pack(side="left", padx=(0, 10), pady=5)

    entry_train_path = ctk.CTkEntry(frame_train_path, placeholder_text="Select training folder")
    entry_train_path.insert(0, app.config['paths']['train'])
    entry_train_path.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=5)

    button_browse_train = ctk.CTkButton(
        frame_train_path, 
        text="Browse", 
        command=lambda: _browse_folder('train', entry_train_path)
    )
    button_browse_train.pack(side="left", pady=5)  

    # Validation
    frame_valid_path = ctk.CTkFrame(content_frame, fg_color='transparent')
    frame_valid_path.pack(fill='x', padx=10, pady=5)

    label_valid_path = ctk.CTkLabel(frame_valid_path, text='Valid Folder:')
    label_valid_path.pack(side='left', padx=(0, 10), pady=5)

    entry_valid_path = ctk.CTkEntry(frame_valid_path, placeholder_text="Select validation folder")
    entry_valid_path.insert(0, app.config['paths']['valid'])
    entry_valid_path.pack(side='left', fill='x', expand=True, padx=(0, 10), pady=5)

    button_browse_valid = ctk.CTkButton(
        frame_valid_path, 
        text='Browse', 
        command=lambda: _browse_folder('valid', entry_valid_path)
    )
    button_browse_valid.pack(side='left', pady=5)

    # Build configuration widget and get the selected config var
    selected_config_var = build_config_widget(content_frame, app)

    # Training buttons
    frame_bottom = ctk.CTkFrame(content_frame, fg_color='transparent')
    frame_bottom.pack(side="left", fill="both", expand=True, padx=10, pady=5)
    
    label_model_name = ctk.CTkLabel(frame_bottom, text='Model Name:')
    label_model_name.pack(side='left', padx=(0, 10), pady=5)
    entry_model_name = ctk.CTkEntry(frame_bottom, placeholder_text="Enter model name")
    entry_model_name.insert(0, app.config['name'])
    entry_model_name.pack(side='left', fill="x", expand=True, padx=(0, 10), pady=5)
    
    button_train_position = ctk.CTkButton(
        frame_bottom, 
        text="Train Position", 
        command=lambda: app.start_training(selected_config_var, entry_model_name.get(), 'position')
    )
    button_train_position.pack(side='left', padx=10, pady=5)
    app.button_train_position = button_train_position

    button_train_keypress = ctk.CTkButton(
        frame_bottom, 
        text='Train Keypress', 
        command=lambda: app.start_training(selected_config_var, entry_model_name.get(), 'keypress')
    )
    button_train_keypress.pack(side='left', padx=10, pady=5)
    app.button_train_keypress = button_train_keypress
    
    # Log frame
    log_frame = ctk.CTkFrame(parent, fg_color='transparent')
    log_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
    
    # Textbox for logging training printouts on the right
    train_log = ctk.CTkTextbox(log_frame, width=775, height=150)
    train_log.pack(fill="both", expand=True)
    train_log.configure(state='disabled')
    # app.write_to_log(train_log, 'Hello, outputs will go here :)')
    
    app.train_log = train_log
    _get_folder_contents('train', app.config['paths']['train'])
    _get_folder_contents('valid', app.config['paths']['valid'])