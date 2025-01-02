import customtkinter as ctk
from tkinter import filedialog
from gui.widget_song_select import build_song_select_widget

global_app = None

"""
Function to load .pth model file from path then displays hyperparameters
Updates app.config['models'] with selected model path
"""
def _select_model(type, entry, frame):
    model_path = filedialog.askopenfilename(
        title=f'Select {type} Model File',
        filetypes=[('PyTorch Model Files', '*.pth'), ('All Files', '*.*')]
    )
    if model_path:
        entry.delete(0, 'end')
        entry.insert(0, model_path)
        config = global_app.config['models']
        config[type] = model_path
        global_app.save_config('models', config)
        global_app.load_model(type)
        
        if type == "Position":
            _display_hyperparams(frame, global_app.pos_hyperparams)
        else:
            _display_hyperparams(frame, global_app.key_hyperparams)

"""
Displays the loaded model hyperparameters
"""
def _display_hyperparams(frame, hps):
    for widget in frame.winfo_children():
        widget.destroy()
    
    # Create a label for each hyperparameter and it's value
    for param, value in hps.items():
        if 'buzz' not in param.lower():
            hp_label = ctk.CTkLabel(frame, text=f'{param}: {value}')
            hp_label.pack(side='left', padx=10, pady=2)

"""
Inference tab building function
Displays:
Position model path
    |__Position model hyperparameters
Keypress model path
    |__Keypress model hyperparameters
Progress bars
    |__Position progress
    |__Keypress progress
Generate Replay button
Song select widget
"""
def build_inference_tab(parent, app):
    global global_app
    global_app = app
    
    content_frame = ctk.CTkFrame(parent)
    content_frame.pack(side='left', fill='both', padx=10, pady=10)
    
    # Position Model
    frame_pos_model = ctk.CTkFrame(content_frame, fg_color='transparent')
    frame_pos_model.pack(fill='x', padx=10, pady=5)
    
    label_pos_model = ctk.CTkLabel(frame_pos_model, text='Position Model:')
    label_pos_model.pack(side='left', padx=(0, 10), pady=5)
    
    entry_pos_model = ctk.CTkEntry(frame_pos_model, placeholder_text='Select Position Model')
    entry_pos_model.insert(0, app.config['models']['Position'])
    entry_pos_model.pack(side='left', fill='x', expand=True, padx=(0, 10), pady=5)
    
    button_browse_pos_model = ctk.CTkButton(
        frame_pos_model,
        text='Select',
        command=lambda: _select_model('Position', entry_pos_model, frame_pos_hps)
    )
    button_browse_pos_model.pack(side='left', pady=5)
    
    # Position Model Hyperparameters
    frame_pos_hps = ctk.CTkFrame(content_frame)
    frame_pos_hps.pack(fill='x', padx=10, pady=5)
    
    # Keypress Model
    frame_key_model = ctk.CTkFrame(content_frame, fg_color='transparent')
    frame_key_model.pack(fill='x', padx=10, pady=5)
    
    label_key_model = ctk.CTkLabel(frame_key_model, text='Keypress Model:')
    label_key_model.pack(side='left', padx=(0, 10), pady=5)
    
    entry_key_model = ctk.CTkEntry(frame_key_model, placeholder_text='Select Keypress Model')
    entry_key_model.insert(0, app.config['models']['Keypress'])
    entry_key_model.pack(side='left', fill='x', expand=True, padx=(0, 10), pady=5)
    
    button_browse_key_model = ctk.CTkButton(
        frame_key_model,
        text='Select',
        command=lambda: _select_model('Keypress', entry_key_model, frame_key_hps)
    )
    button_browse_key_model.pack(side='left', pady=5)
    
    # Keypress Model Hyperparameters
    frame_key_hps = ctk.CTkFrame(content_frame)
    frame_key_hps.pack(fill='x', padx=10, pady=5)
    
    app.load_model("Position")
    if app.pos_model:
        _display_hyperparams(frame_pos_hps, app.pos_hyperparams)
    app.load_model("Keypress")
    if app.key_model:
        _display_hyperparams(frame_key_hps, app.key_hyperparams)
    
    # Progress bars
    frame_progress_bars = ctk.CTkFrame(content_frame)
    frame_progress_bars.pack(fill='x', padx=10, pady=5)
    frame_progress_bars.pack_forget()
    
    # Position progress
    frame_pos_progress = ctk.CTkFrame(frame_progress_bars, fg_color='transparent')
    frame_pos_progress.pack(fill='x', padx=10, pady=5)
    label_pos_progess = ctk.CTkLabel(frame_pos_progress, text='Position', width=60, anchor='w')
    label_pos_progess.pack(side='left', padx=(0,10), pady=5)
    progress_bar_pos = ctk.CTkProgressBar(frame_pos_progress, orientation='horizontal')
    progress_bar_pos.pack(fill='x', padx=10, pady=15)
    progress_bar_pos.set(0)
    
    # Keypress progress
    frame_key_progress =ctk.CTkFrame(frame_progress_bars, fg_color='transparent')
    frame_key_progress.pack(fill='x', padx=10, pady=5)
    label_key_progress=ctk.CTkLabel(frame_key_progress, text='Keypress', width=60, anchor='w')
    label_key_progress.pack(side='left', padx= (0, 10), pady=5)
    progress_bar_key = ctk.CTkProgressBar(frame_key_progress, orientation='horizontal')
    progress_bar_key.pack(fill='x', padx=10, pady=15)
    progress_bar_key.set(0)
    
    app.pb_pos = progress_bar_pos
    app.pb_key = progress_bar_key
    app.frame_pb = frame_progress_bars
    
    # Generate replay button
    button_generate_replay = ctk.CTkButton(
        content_frame,
        text='Generate Replay',
        command=lambda: app.generate_replay(),
        state='disabled'
    )
    button_generate_replay.pack(side='bottom', pady=5)
    app.button_generate_replay = button_generate_replay
    
    # Song select widget on the right
    build_song_select_widget(parent, app)