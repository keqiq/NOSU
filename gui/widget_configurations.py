import tkinter as tk
import customtkinter as ctk

global_app = None

"""
Function for creating a new configuration
Saves new configuration to app.config['train'] with empty configuration
"""
def _create_new_config(configs, frame_configs, frame_params, selected_config_var=None):
    dialog = ctk.CTkToplevel() 
    dialog.title("New Configuration")
    dialog.geometry("300x150")

    ctk.CTkLabel(dialog, text="Enter Configuration Name:", font=("Arial", 14)).pack(pady=10)

    entry_name = ctk.CTkEntry(dialog, placeholder_text="Configuration Name")
    entry_name.pack(pady=10)

    def save_new_config():
        config_name = entry_name.get().strip()
        if not config_name:
            ctk.CTkLabel(dialog, text="Name cannot be empty!", text_color="red").pack()
            return
        if config_name in configs:
            ctk.CTkLabel(dialog, text="Name already exists!", text_color="red").pack()
            return

        configs[config_name] = {}
        global_app.save_config('train', configs)
        global_app.write_to_log(global_app.train_log, f'Created new config {config_name}')
        _display_configs(configs, frame_configs, frame_params, selected_config_name=config_name, selected_config_var=selected_config_var)
        dialog.destroy()

    ctk.CTkButton(dialog, text="Save", command=save_new_config).pack(pady=10)

"""
Function to store values in parameter entries into app.config['train'][config_name]
Only values which differ from default configuration will be written to config
"""
def _save_parameters(config, config_name, entries):
    # Ensure config_name is present
    if config_name not in config:
        config[config_name] = {}

    for key, entry in entries.items():
        section, param = key.split('.')
        value_str = entry.get().strip()
        
        # Attempt type conversion: int, float, or fallback to string
        if value_str.isdigit():
            value = int(value_str)
        else:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str

        # Get the default value for comparison
        default_value = config['default'][section][param]

        # If the value matches the default, remove it from the config, if it exists
        if value == default_value:
            if section in config[config_name] and param in config[config_name][section]:
                del config[config_name][section][param]
                # If the section is now empty, remove the section
                if not config[config_name][section]:
                    del config[config_name][section]
        else:
            # The value differs from default; store it
            if section not in config[config_name]:
                config[config_name][section] = {}
            config[config_name][section][param] = value

    global_app.save_config('train', config)
    global_app.write_to_log(global_app.train_log, f'Saved configuration {config_name}')

"""
Function to display configuration parameter values in side entries
Does not display default parameters and does not allow save over default parameters
"""
def _display_parameters(config, config_name, frame):
    for widget in frame.winfo_children():
        widget.destroy()
    
    if config_name == 'default':
        return
    
    # Containers for parameter sections
    frames = {}
    entries = {}
    for section in config['default'].keys():
        section_frame = ctk.CTkFrame(frame)
        section_frame.pack(fill='x', padx=10, pady=5)
        ctk.CTkLabel(section_frame, text=section, font=('Arial', 16, 'bold')).pack(anchor='w', padx=5, pady=5)
        frames[section] = section_frame
        
    for section, default_params in config['default'].items():
        section_frame = frames[section]
        
        # Check for parameters in custom config, fallback to default
        custom_params = config.get(config_name, {}).get(section, {})
        row = 0
        col = 0
        for param, default_value in default_params.items():
            value = custom_params.get(param, default_value)
            
            if col == 0:
                row_frame = ctk.CTkFrame(section_frame)
                row_frame.pack(fill='x', padx=5, pady=5)
                
            # Parameter label
            label = ctk.CTkLabel(row_frame, text=param, width=150, anchor='e')
            label.grid(row=row, column=col * 2, padx=5, pady=5, sticky='w')
            
            # Parameter value textbox
            entry = ctk.CTkEntry(row_frame, width=60)
            entry.insert(0, str(value))
            entry.grid(row=row, column=col*2+1, padx=(0, 10), pady=5, sticky='w')
            entries[f'{section}.{param}'] = entry
            
            col += 1
            if col >= 3:
                col = 0
                row += 1
                
    save_button = ctk.CTkButton(frame, 
                                text='Save', 
                                command=lambda c=config, cn=config_name, e=entries:
                                    _save_parameters(c, cn, e))
    save_button.pack(pady=10)

"""
Function to display all configurations inside app.config['train'] in a row
Clicking on a configuration will call _display_parameters to display that configuration's parameters
Places a plus button to call _create_new_config
"""
def _display_configs(configs, frame_configs, frame_params, selected_config_name=None, selected_config_var=None):
    selected_button = {"current": None}
    buttons = {}  # store references to buttons

    def create_button_action(configs, config_name, frame_params, button):
        def select_config():
            if selected_button["current"]:
                selected_button["current"].configure(fg_color=global_app.unselected_color)
            button.configure(fg_color=global_app.selected_color)
            selected_button["current"] = button
            _display_parameters(configs, config_name, frame_params)
            if selected_config_var is not None:
                selected_config_var.set(config_name)
        return select_config

    # Refresh all widget on update
    for widget in frame_configs.winfo_children():
        widget.destroy()

    # Create buttons with config names
    for config_name in configs.keys():
        button = ctk.CTkButton(
            frame_configs,
            width=100,
            text=config_name,
            fg_color=global_app.unselected_color
        )
        button.pack(side='left', padx=5)
        button.configure(command=create_button_action(configs, config_name, frame_params, button))
        buttons[config_name] = button

    # Plus button to create new config
    plus_button = ctk.CTkButton(
        frame_configs,
        width=35,
        text="+",
        command=lambda: _create_new_config(configs, frame_configs, frame_params, selected_config_var)
    )
    plus_button.pack(side='left', padx=5)

    if selected_config_name and selected_config_name in buttons:
        buttons[selected_config_name].invoke()

"""
Configuration widget building function
Displays:
Configs
    |__default
    |__custom
    ...
Parameters
    |__Data
        |__Data parameters
        ...
    |__Model
        |__Model parameters
        ...
    |__Loss
        |__Loss parameters
        ...
    |__Train
        |__Train parameters
        ...
Save button
"""
def build_config_widget(parent, app):
    global global_app
    global_app = app
    configurations = app.config
    selected_config_var = tk.StringVar()  # To store the currently selected config

    frame_configs = ctk.CTkFrame(parent, fg_color='transparent')
    frame_configs.pack(fill='x', padx=10, pady=10)
    frame_params = ctk.CTkFrame(parent, fg_color='transparent')
    frame_params.pack(fill='both', expand=True, padx=10, pady=10)

    # Default to selecting 'default'
    _display_configs(configurations['train'], frame_configs, frame_params, selected_config_name='default', selected_config_var=selected_config_var)

    return selected_config_var