import numpy as np
import time
import pygame
import sys

# replay parameters

WIDTH, HEIGHT = 512, 384

FPS = 60

BACKGROUND = (0, 0, 0)

HIT_CIRCLE_RADIUS = 30

HIT_CIRCLE_COLOR = (255, 255, 255)

SLIDER_RADIUS = 20

SLIDER_COLOR = (128, 128, 255)

SPINNER_RADIUS = 40

SPINNER_COLOR = (255, 128, 128)

NOTE_FRAME_DURATION = FPS / 2

REACTION_TIME = ((1000 * NOTE_FRAME_DURATION)/ FPS)


CURSOR_RADIUS = 5

CURSOR_COLOR = {
    #0
    0 : (128, 128, 128),

    #K1|M1
    1 : (255, 0, 0),
    5 : (255, 0, 0),

    #K2|M2
    2 : (0, 255, 0),
    10 : (0, 255, 0),

    #K1|K2|M1|M2
    3: (255, 255, 0),
    11 : (255, 255, 0),
    15 : (255, 255, 0),

    #SMOKE
    21: (255, 0, 255)
}

def render_hit_circle(hc, elapsed_time, screen):
    ttl = max(0, (hc[2] - elapsed_time)) / REACTION_TIME
    cicle_radius = HIT_CIRCLE_RADIUS - 10 * (1 - ttl)
    alpha_surface = pygame.Surface((HIT_CIRCLE_RADIUS*2, HIT_CIRCLE_RADIUS*2), pygame.SRCALPHA)
    pygame.draw.circle(alpha_surface, HIT_CIRCLE_COLOR, (HIT_CIRCLE_RADIUS, HIT_CIRCLE_RADIUS), cicle_radius)
    alpha_surface.set_alpha(min((1-ttl) * 255, 255))
    screen.blit(alpha_surface, (hc[0] - HIT_CIRCLE_RADIUS, hc[1] - HIT_CIRCLE_RADIUS))
    
def render_slider(slider, time, screen):
    points = slider[: ,:2]
    if len(points) < 2: points = np.array([points[0], points[0]])
    pygame.draw.lines(screen, (255, 255, 255), False, points, 5)

    splice = slider[:, 2]
    last_tick_index = np.searchsorted(splice, time, side='right') - 1
    if last_tick_index >= 0:
        last_tick = slider[last_tick_index]
    else:
        last_tick = slider[0]
        
    pygame.draw.circle(screen, SLIDER_COLOR, (last_tick[0], last_tick[1]), SLIDER_RADIUS)
    
def render_spinner(obj, screen):
    pygame.draw.circle(screen, SPINNER_COLOR, (obj[0], obj[1]), SPINNER_RADIUS)
    

def visualize(map, replay, song):
    hit_circles = map['hit_circles']
    sliders = map['sliders']
    spinners = map['spinners']
    
    hc_timings = hit_circles[:, 2]
    if (len(sliders) > 0):
        sld_timings = np.array([[slider[0, 2], slider[0, 4]] for slider in sliders])
    if (len(spinners) > 0):
        sp_timings = spinners[:, [2, 4]]
    input_timings = replay[:, 2]
    
    pygame.init()
    pygame.mixer.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()

    # Playing song
    track = pygame.mixer.Sound(song)
    track.play()
    time.sleep(0.5)
    
    #Game loop
    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(BACKGROUND)

        elapsed_time = pygame.time.get_ticks() - start_time

        # Rendering hit circles
        hc_mask = (hc_timings >= elapsed_time) & (hc_timings <= elapsed_time + REACTION_TIME) 
        for hit_circle in hit_circles[hc_mask]:
            render_hit_circle(hit_circle, elapsed_time, screen)

        # Rendering sliders
        if (len(sliders) > 0):
            sld_mask = (sld_timings[:, 0] <= elapsed_time + REACTION_TIME) & (sld_timings[:, 1] >= elapsed_time)
            for slider in sliders[sld_mask]:
                render_slider(slider, elapsed_time, screen)
        
        # Rendering spinners - i think only 1 spinner is on screen at a time but just in case
        if (len(spinners) > 0):
            sp_mask = (sp_timings[:, 0] <= elapsed_time + REACTION_TIME) & (sp_timings[:, 1] >= elapsed_time)
            for spinner in spinners[sp_mask]:
                render_spinner(spinner, screen)
            
        # Displaying cursor
        last_input_index = np.searchsorted(input_timings, elapsed_time, side='right') - 1
        last_input = replay[last_input_index]
        pygame.draw.circle(screen, CURSOR_COLOR[last_input[3]], (last_input[0], last_input[1]), CURSOR_RADIUS)
   
        # Update frame
        pygame.display.flip()

        clock.tick(FPS)

                