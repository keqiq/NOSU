from osrparse import Replay

from .map.HitObjects import HitObjects
from .map.MapParams import MapParams
from .map.TimingPoints import TimingPoints

import numpy as np

def parse_map(map_path, hr):      
    with open(map_path, 'r', encoding='utf-8') as file:
            lines = file.read()
    
    sections = lines.split('\n\n')

    for section in sections:
        content = section.split('\n')
        header = content.pop(0)
        if header == '': header = content.pop(0)

        if header == '[Difficulty]':
            mp = MapParams(content)

        elif header == '[TimingPoints]':
            tp = TimingPoints(content)

        elif header == '[HitObjects]':
            if content[-1] == '': content.pop(-1)
            ho = HitObjects(mp, tp, content, hr)
    
    return ho.get_data()


def parse_replay_pos(replay_path):
    replay = Replay.from_path(replay_path)
    replay_data = replay.replay_data
    
    # Check for hard rock, this mod changes the positions of hit objects
    hr = False
    mods_flag = replay.mods.value
    if ((mods_flag >> 4) & 1):
        hr = True
    results = []
    time = replay_data[1].time_delta
    
    i = 2
    while i < len(replay_data):
        time_delta = replay_data[i].time_delta
        
        # Some samples intervals are irregular < 16ms due to keypress inputs
        # This will merge those samples so that most intervals are 16 or 17 ms apart
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

def parse_replay_key(replay_path):
    replay = Replay.from_path(replay_path)
    replay_data = replay.replay_data
    results = []
    time = replay_data[1].time_delta

    for event in replay_data[2:]:
        time += event.time_delta
        results.append([event.x, event.y, time, event.keys.value])
    
    return np.array(results)