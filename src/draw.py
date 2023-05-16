import pygame
import sys
from pygame.locals import QUIT, MOUSEBUTTONDOWN, KEYDOWN

def save_track(tracks_dict):
    with open('track.txt', 'w') as f:
        for track_type, segments in tracks_dict.items():
            f.write(f'{track_type}:\n')
            for segment in segments:
                for line in segment:
                    f.write(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}\n')
            f.write('\n')

pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Draw Track')
clock = pygame.time.Clock()

WALL_MODE = "WALLS"
CHECKPOINT_MODE = "CHECKPOINTS"
drawing_mode = WALL_MODE

tracks = {
    WALL_MODE: [[]],
    CHECKPOINT_MODE: [[]]
}
points = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()

            if len(points) >= 1:
                tracks[drawing_mode][-1].append(points[-1] + pos)
            points.append(pos)

        if event.type == KEYDOWN:
            if event.key == pygame.K_s:
                save_track(tracks)
                print('Track saved as track.txt')
            elif event.key == pygame.K_u:
                if len(tracks[drawing_mode][-1]) > 0:
                    tracks[drawing_mode][-1].pop()
                    points.pop()
                    print('Last point removed')
            elif event.key == pygame.K_n:
                tracks[drawing_mode].append([])
                points = []
                print('New segment started')
            elif event.key == pygame.K_r:
                tracks[drawing_mode] = [[]]
                points = []
                print('New segment started')
            elif event.key == pygame.K_1:
                drawing_mode = WALL_MODE
                print('Switched to Wall mode')
            elif event.key == pygame.K_2:
                drawing_mode = CHECKPOINT_MODE
                print('Switched to Checkpoint mode')

    screen.fill((255, 255, 255))

    for track_type, segments in tracks.items():
        color = (0, 0, 255) if track_type == CHECKPOINT_MODE else (0, 0, 0)
        for segment in segments:
            for line in segment:
                pygame.draw.line(screen, color, line[:2], line[2:], 5)

    clock.tick(60)
    pygame.display.flip()
