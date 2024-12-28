import customtkinter as ctk
from tkinter import filedialog
import os
global_app = None

"""
Function for selecting osu!'s song directory
Saves the selected song directory to app.config['song_path']
"""
def _browse_song_folder(entry_song_path):
    folder_selected = filedialog.askdirectory(title=f'Select osu! Songs folder', initialdir=global_app.song_path)
    if folder_selected:
        entry_song_path.delete(0, 'end')
        entry_song_path.insert(0, folder_selected)
        
    config = global_app.config['song_path']
    config = folder_selected
    global_app.save_config('song_path', config)

"""
Function to update app.selected_map
If the selected map is standard mode, enable generate replay button
"""
def _select_map_file(map_file, map_path):
    global_app.process_map_file(os.path.join(map_path, map_file))

    # Highlight the selected map file and reset others
    for btn_map in global_app.map_buttons.values():
        btn_map.configure(fg_color=global_app.unselected_color)  # Reset all map colors
    global_app.map_buttons[map_file].configure(fg_color=global_app.selected_color)  # Highlight selected map
    global_app.update_buttons()

"""
Function to display .osu map files inside of a song folder as buttons on the right
On click, the button will call _select_map_file
"""
def _load_map_files(folder, frame_beatmap_list):
    global_app.selected_map = None
    for widget in frame_beatmap_list.winfo_children():
        widget.destroy()

    # Update the selected folder style
    for btn_folder in global_app.folder_buttons.values():
        btn_folder.configure(fg_color=global_app.unselected_color)  # Reset all folder colors
    global_app.folder_buttons[folder].configure(fg_color=global_app.selected_color)  # Highlight selected folder

    map_path = os.path.join(global_app.song_path, folder)
    map_files = [file for file in os.listdir(map_path) if file.endswith('.osu')]

    # Check if there are any .osu files
    if not map_files:
        no_files_label = ctk.CTkLabel(frame_beatmap_list, text='No map files found', anchor='w')
        no_files_label.pack(fill='x', padx=5, pady=2)
    else:
        global_app.map_buttons = {}  # Store references to map file buttons
        for map_file in sorted(map_files):
            map_button = ctk.CTkButton(
                frame_beatmap_list,
                text=map_file,
                command=lambda f=map_file: _select_map_file(f, map_path),
                anchor='w',
                fg_color=global_app.unselected_color
            )
            map_button.pack(fill='x', padx=5, pady=2)
            global_app.map_buttons[map_file] = map_button
            
    global_app.update_buttons()

"""
Function to display all song folders in osu!'s song directory as buttons on the left
On click, call _load_map_files to display the songs in the folder
Also filter out folders which do not match search_query
"""
def _load_song_folders(frame_song_list, frame_beatmap_list, search_query=""):
    for widget in frame_song_list.winfo_children():
        widget.destroy()

    # Load all folders from the osu! songs directory
    all_folders = [
        folder for folder in os.listdir(global_app.song_path)
        if os.path.isdir(os.path.join(global_app.song_path, folder))
    ]

    # Keep folders which have name that contain 'search_query'
    # Matching is case-insensitive
    if search_query:
        search_query_lower = search_query.lower()
        song_folders = [f for f in all_folders if search_query_lower in f.lower()]
    else:
        song_folders = all_folders

    global_app.folder_buttons = {}  # Store references to folder buttons
    for folder in sorted(song_folders):
        folder_button = ctk.CTkButton(
            frame_song_list,
            text=folder,
            command=lambda f=folder: _load_map_files(f, frame_beatmap_list),
            anchor='w',
            fg_color=global_app.unselected_color,
        )
        folder_button.pack(fill='x', padx=5, pady=2)
        global_app.folder_buttons[folder] = folder_button  # Store the reference
        
"""
Song select widget building function
Displays:
Song path
Song browser
    |__Song list    (left)
        |__Search bar
        |__Song_1
        ...
        |__Song_n
    |__Beatmap list (right)
        |__Beatmap_1
        ...
        |__Beatmap_n
"""
def build_song_select_widget(parent, app):
    global global_app
    global_app = app
    
    content_frame = ctk.CTkFrame(parent)
    content_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Song path
    frame_song_path = ctk.CTkFrame(content_frame, fg_color='transparent')
    frame_song_path.pack(fill='x', padx=10, pady=5)
    
    label_song_path = ctk.CTkLabel(frame_song_path, text='osu! Songs Folder:')
    label_song_path.pack(side='left', padx=10, pady=5)
    
    entry_song_path = ctk.CTkEntry(frame_song_path, placeholder_text="Select osu! Songs folder")
    entry_song_path.insert(0, app.song_path)
    entry_song_path.pack(side='left', fill='x', expand=True, padx=(0, 10), pady=5)
    
    button_song_path = ctk.CTkButton(
        frame_song_path,
        text='Browse',
        command=lambda: _browse_song_folder(entry_song_path)
    )
    button_song_path.pack(side='left', pady=5)
    
    # Song browser frame
    frame_song_browser = ctk.CTkFrame(content_frame, fg_color='transparent')
    frame_song_browser.pack(fill='both', expand=True)
    
    frame_song_browser.columnconfigure(0, weight=1)  # Left frame: Song list
    frame_song_browser.columnconfigure(1, weight=2)  # Right frame: Beatmap list
    frame_song_browser.rowconfigure(0, weight=1)     # Make the row stretch vertically
    
    # Song folder list (left)
    frame_left = ctk.CTkFrame(frame_song_browser, fg_color='transparent')
    frame_left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

    # Search bar
    frame_search = ctk.CTkFrame(frame_left, fg_color='transparent')
    frame_search.pack(fill='x', padx=10, pady=5)

    entry_search = ctk.CTkEntry(frame_search, placeholder_text="Search songs...")
    entry_search.pack(side='left', fill='x', expand=True, padx=(0, 10))

    # This will trigger whenever the user modifies the search text
    entry_search.bind(
        "<KeyRelease>",
        lambda event: _load_song_folders(frame_song_list, frame_beatmap_list, search_query=entry_search.get())
    )

    frame_song_list = ctk.CTkScrollableFrame(frame_left, fg_color='transparent')
    frame_song_list.pack(fill='both', expand=True)
    
    # Beatmap list (right)
    frame_right = ctk.CTkFrame(frame_song_browser, fg_color='transparent')
    frame_right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
    
    frame_beatmap_list = ctk.CTkScrollableFrame(frame_right, fg_color='transparent')
    frame_beatmap_list.pack(fill='both', expand=True)
    
    # Load the folders with no search query:
    _load_song_folders(frame_song_list, frame_beatmap_list, search_query="")