import pygame, sys, math, torch, random
import numpy as np

from neuralnetwork import PPO
from raytracer import RayTracer
from collections import deque
from pygame.locals import QUIT

class Track:
  def __init__(self, file_name):
    self.walls = []
    self.checkpoints = []
    self.current_mode = None

    with open(file_name, 'r') as f:
      for line in f:
        line = line.strip()
        if not line:
          continue

        if line.endswith(':'):
          self.current_mode = line[:-1]
        else:
          x1, y1, x2, y2 = [int(i) for i in line.split(', ')]
          if self.current_mode == 'WALLS':
            self.walls.append((x1, y1, x2, y2))
          elif self.current_mode == 'CHECKPOINTS':
            self.checkpoints.append((x1, y1, x2, y2))

class Line:
  def __init__(self, x1, y1, x2, y2):
    self.points = [x1, y1, x2, y2]
    self.p1 = [x1, y1]
    self.p2 = [x2, y2]
  def __repr__(self):
    return f'Line({self.points})'
  
  def length(self):
    return ((self.p1[0]-self.p2[0])**2+(self.p1[1]-self.p2[1])**2)**0.5
  
  def intersects(self, line2):
    x1, y1, x2, y2 = self.points
    x3, y3, x4, y4 = line2.points
    den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if den == 0:
      return False
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
    if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and
      min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
      return True
    return False

class Car: 
  def __init__(self, pos = [200,55], rot=-90, ID='0', color='#f00', weights=[], biases=[]):
    self.pos = pos
    self.rot = rot
    self.ID = ID
    self.color = color
    self.weights = weights
    self.biases = biases
    self.vel = [0,0]
    self.accel = .2
    self.speed = 4.0
    self.width = 6
    self.height = 10
    self.rayTracer = RayTracer(pos, rot, 250, (-fov//2, fov//2), fov//tracers)

    self.score = 0
    self.age = 0.0
    self.time_since_checkpoint = 0.0

  def reset(self):
    self = self.initial.copy()
  
  def intersects(self, line):
    vertices = self.vertices()
    for i in range(len(vertices)):
      car_line = (vertices[i][0],vertices[i][1],vertices[i-1][0],vertices[i-1][1])
      if line.intersects(Line(*car_line)):
        return True
    return False
  
  def vertices(self):
    sin = math.sin(math.radians(self.rot))
    cos = math.cos(math.radians(self.rot))
    return [
      [self.pos[0] + sin*self.height/2 - cos*self.width/2,
      self.pos[1] + cos*self.height/2 + sin*self.width/2],
      [self.pos[0] - sin*self.height/2 - cos*self.width/2,
      self.pos[1] - cos*self.height/2 + sin*self.width/2],
      [self.pos[0] - sin*self.height/2 + cos*self.width/2,
      self.pos[1] - cos*self.height/2 - sin*self.width/2],
      [self.pos[0] + sin*self.height/2 + cos*self.width/2,
      self.pos[1] + cos*self.height/2 - sin*self.width/2],
    ]
  
  def collides(self, track):
    for line in track:
      if self.intersects(Line(*line)):
        return True
    return False

  def choose_action(action_probs):
      return np.random.choice(5, p=action_probs)

  def cap_velocity(velocity, max_speed=1.5):
      magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2)
      if magnitude > max_speed:
          normalized_velocity = [component / magnitude for component in velocity]
          return [component * max_speed for component in normalized_velocity]
      else: return velocity

  def passed_checkpoint(self, checkpoint_line):
    for i in range(4):
      if self.intersects(checkpoint_line):
        return True
    return False

  def calculate_reward(self):
    checkpoint_reward = 10
    time_penalty = 2.5

    reward = checkpoint_reward * self.score \
          + time_penalty * self.age
    
    return reward
  
  def render(self, screen):
    sin = math.sin(math.radians(self.rot[0]))
    cos = math.cos(math.radians(self.rot[0]))

    vertices = [
      [self.pos[0] + sin*self.height/2 - cos*self.width/2,
      self.pos[1] + cos*self.height/2 + sin*self.width/2],
      [self.pos[0] - sin*self.height/2 - cos*self.width/2,
      self.pos[1] - cos*self.height/2 + sin*self.width/2],
      [self.pos[0] - sin*self.height/2 + cos*self.width/2,
      self.pos[1] - cos*self.height/2 - sin*self.width/2],
      [self.pos[0] + sin*self.height/2 + cos*self.width/2,
      self.pos[1] + cos*self.height/2 - sin*self.width/2],
    ]

    pygame.draw.polygon(screen, 'red', vertices)

pi = 3.141592653589793
score = 0
FPS = 30
multiplier = 10
tracers = 8
fov = 360
spawn = [200,55]
width,height = 800,600

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('vroom!')
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 10)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# nn = PPO(11, 64, 5, device)

track = Track('tracks/new.track')

epoch = 0
cars = np.empty(10, dtype = Car)
running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT: running = False
  pygame.draw.rect(screen, (255,255,255), pygame.Rect(0, 0, width, height))
  tick = clock.tick(FPS)
  dt = tick / (1000/(FPS*multiplier))

  


  pygame.display.update()

'''
time_since_checkpoint += (tick/1000) * multiplier
time_elapsed += (tick/1000) * multiplier
for event in pygame.event.get():
  if event.type == QUIT:
    pygame.quit()
    sys.exit()
  if event.type == pygame.KEYDOWN:
    keys.append(event.key)
  if event.type == pygame.KEYUP:
    keys.remove(event.key)

# Update and display RayTracer
ray_tracer.update()
# track is a list of lines. lines look like [x1,y1,x2,y2]
ray_tracer.display(screen,track.walls)

# Collect input data for the neural network
nn_input = []
for i in ray_tracer.distances:
  nn_input.append(i)
nn_input.append(vel[0])
nn_input.append(vel[1])
nn_input.append(rot[0])

# Predict the output using the neural network
predicted_output = nn.calculate_outputs(nn_input)

# Map the predicted output to actions
action = choose_action(predicted_output)

# Perform the corresponding action
if action == 0:  # Move forward
  vel[0] -= math.sin(math.radians(rot[0]))*accel*dt
  vel[1] -= math.cos(math.radians(rot[0]))*accel*dt
elif action == 1:  # Turn left
  rot[0] += .8*dt
elif action == 2:  # Turn right
  rot[0] -= .8*dt
elif action == 3: # Brake
  vel[0] = vel[0]*.9
  vel[1] = vel[1]*.9

vel = cap_velocity(vel,3)
pos[0]+=vel[0]*dt
pos[1]+=vel[1]*dt
#vel = [*map(lambda i:.99*i if abs(i)>.1 else 0, vel),]
# this shit caused issues. idk. friction no worky


coords = font.render(f'[{pos[0]//1},{pos[1]//1}], {rot[0]}*', True, (255, 0, 0))
screen.blit(coords, (5, 5))
# draw track lines
for line in track.walls:
  pygame.draw.line(screen, 'red', (line[0], line[1]),
                    (line[2], line[3]), 2)
# draw checkpoint lines
for i,line in enumerate(track.checkpoints):
  pygame.draw.line(screen,'blue', (line[0], line[1]),
                    (line[2], line[3]), 2)
  # draw checkpoint number
  num = font.render(str(i), True, (0, 0, 0))
  screen.blit(num, ((line[0] + line[2]) / 2, (line[1] + line[3]) / 2))

  # Added check for passing the checkpoint and increase the score
if len(track.checkpoints) > 0:
  if detect_collision(vertices, [track.checkpoints[score % len(track.checkpoints)]]):
    score += 1
    time_since_checkpoint = 0

  # Check for collision and reset the car if there's a collision
if detect_collision(vertices, track.walls) or time_since_checkpoint >= 15:
  time_elapsed = time_since_checkpoint = 0 
  pos, rot, vel, nn = reset_car(pos, rot, vel, [11, 9, 5])
  ray_tracer = RayTracer(pos, rot, 250, (-fov//2, fov//2), fov//tracers)

  reward = 0

# Score display
score_text = font.render(f'Score: {score}', True, (0, 0, 0))
screen.blit(score_text, (width - 70, 5))
time_text = font.render(str(time_since_checkpoint), True, (0, 0, 0))
screen.blit(time_text, (width - 70, 15))

s = font.render(str((vel[0]**2+vel[1]**2)**.5), True, 'blue')
screen.blit(s, (5,15))
for i,d in enumerate(ray_tracer.distances):
  t = font.render(str(round(d,2)), True, 'blue')
  screen.blit(t, (5,25+i*10))

# Draw velocity vector
velocity_scale = 20  # Adjust this value to change the length of the velocity vector
velocity_vector = (pos[0] + vel[0] * velocity_scale, pos[1] + vel[1] * velocity_scale)
pygame.draw.line(screen, (0, 0, 255), (pos[0],pos[1]), velocity_vector, 2)
  '''