import pygame, sys, math, torch, random, os
import numpy as np

from neuralnetwork import PPO
from raytracer import RayTracer
from collections import deque
from pygame.locals import QUIT

def mutate_weights_biases(parent_model, mutation_rate=0.1, mutation_scale=0.5):
    parent_weights = parent_model.get_weights()
    parent_biases = parent_model.get_biases()

    def mutate_array(arr):
        mutation_mask = np.random.rand(*arr.shape) < mutation_rate
        mutation_values = np.random.uniform(-mutation_scale, mutation_scale, arr.shape)
        return arr + (mutation_mask * mutation_values)

    mutated_weights = [mutate_array(w) for w in parent_weights]
    mutated_biases = [mutate_array(b) for b in parent_biases]

    return mutated_weights, mutated_biases

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
            self.walls.append(Line(x1, y1, x2, y2))
          elif self.current_mode == 'CHECKPOINTS':
            self.checkpoints.append(Line(x1, y1, x2, y2))
  def render(self, screen):
    for line in self.walls:
      pygame.draw.line(screen, 'red', (line.points[0], line.points[1]), (line.points[2], line.points[3]), 2)
    for i,line in enumerate(self.checkpoints):
      pygame.draw.line(screen,'blue', (line.points[0], line.points[1]), (line.points[2], line.points[3]), 2)
      num = font.render(str(i), True, (0, 0, 0))
      screen.blit(num, ((line.points[0] + line.points[2]) / 2, (line.points[1] + line.points[3]) / 2))

class Line:
  def __init__(self, x1, y1, x2, y2):
    self.points = [x1, y1, x2, y2]
    self.p1 = [x1, y1]
    self.p2 = [x2, y2]
  def __repr__(self):
    return self.points
  def __str__(self):
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
    self.vel = [0,0]
    self.accel = 1
    self.speed = 6
    self.width = 6
    self.height = 10
    self.rayTracer = RayTracer(pos, rot, 250, (-fov//2, fov//2), fov//tracers)
    self.initial = {
      'pos': pos,
      'rot': rot,
      'vel': [0,0],
      'accel': 1,
      'speed': 6,
      'width': 6,
      'height': 10,
      'rayTracer': {
        'pos': pos,
        'rot': rot,
        'length': 250,
        'angle_range': (-fov//2, fov//2),
        'angle_step': fov//tracers
      }

    }
    self.nn = PPO(11, 64, 5)

    self.score = 0
    self.age = 0.0
    self.time_since_checkpoint = 0.0

  def reset(self):
    for i in self.initial:
      if i == 'rayTracer':
        self.rayTracer = RayTracer(**self.initial[i])
      else:
        setattr(self, i, self.initial[i])
    self.score = 0
    self.age = 0.0
    self.time_since_checkpoint = 0.0
  
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
      if self.intersects(line):
        return True
    return False

  def choose_action(self, action_probs):
    arr = [np.random.choice(2, p=[1-prob, prob]) for prob in action_probs]
    return int(''.join([str(i) for i in arr]), 2)
  

  def cap_velocity(self, velocity, max_speed=1.5):
      magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2)
      if magnitude > max_speed:
          normalized_velocity = [component / magnitude for component in velocity]
          return [component * max_speed for component in normalized_velocity]
      else: return velocity

  def passed_checkpoint(self, checkpoint_lines):
    for l in checkpoint_lines:
      for i in range(4):
        if self.intersects(l):
          return True
    return False

  def calculate_reward(self):
    checkpoint_reward = 10
    time_penalty = 2.5

    reward = checkpoint_reward * self.score \
          + time_penalty * self.age
    
    return reward
  
  def render(self, screen):
    self.rayTracer.display(screen,track.walls)
    pygame.draw.polygon(screen, 'red', car.vertices())
    

  def update(self, tick):
    dt = tick/1000
    self.age += dt
    self.time_since_checkpoint += dt

    self.rayTracer.pos = self.pos
    self.rayTracer.rot = self.rot
    # track is a list of lines. lines look like [x1,y1,x2,y2]
    
    # Collect input data for the neural network
    nn_input = []
    for i in self.rayTracer.distances:
      nn_input.append(i)
    nn_input.append(self.vel[0])
    nn_input.append(self.vel[1])
    nn_input.append(self.rot)

    predicted_output = self.nn.calculate_outputs(nn_input)
    action = self.choose_action(predicted_output)
    
    if action&0b1000>0: # Accelerate
      self.vel[0] -= math.sin(math.radians(self.rot))*self.accel*dt
      self.vel[1] -= math.cos(math.radians(self.rot))*self.accel*dt
    if action&0b0100>0:  # Turn left
      self.rot += 150*dt
    if action&0b0010>0:  # Turn right
      self.rot -= 150*dt
    if action&0b0001>0: # Brake
      self.vel[0] = self.vel[0]*.9
      self.vel[1] = self.vel[1]*.9

    self.vel = self.cap_velocity(self.vel, self.speed)
    self.pos[0] += self.vel[0]
    self.pos[1] += self.vel[1]
    
    if self.passed_checkpoint(track.checkpoints):
      self.score += 1
      self.time_since_checkpoint = 0.0

pi = 3.141592653589793
score = 0
FPS = 60
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
track_file = os.path.join(os.path.dirname(__file__), '../tracks/new.track')
track = Track(track_file)

population_size = 2
cars = [Car() for _ in range(population_size)]

generations = 10
num_best_parents = 1

for generation in range(generations):
    print(f"Generation {generation + 1}")

    for car in cars:
      car.reset()

    running = True
    while running:
      tick = clock.tick(FPS)
      screen.fill('white')
      track.render(screen)
      for event in pygame.event.get():
        if event.type == QUIT:
          pygame.quit()
          sys.exit()
      for i,car in enumerate(cars):
        car.update(tick)
        car.render(screen)

        if car.collides(track.walls) or car.time_since_checkpoint > 15:
          cars.pop(i)
        if len(cars) <= num_best_parents:
          running = False
          break

      # frame rate monitor
      fps = font.render(str(int(clock.get_fps())), True, pygame.Color('Red'))
      screen.blit(fps, (0, 0))
      # cars monitor
      cars_text = font.render(str(len(cars)), True, pygame.Color('Red'))
      screen.blit(cars_text, (0, 10))

      pygame.display.update()

    # Evaluate each car based on its score and age
    cars = sorted(cars, key=lambda car: car.calculate_reward(), reverse=True)

    # Select best parents
    best_parents = cars[:num_best_parents]

    # Create the new generation of cars with mutated weights/biases
    new_generation = []
    for i in range(population_size):
      parent_index = int(best_parents[i % num_best_parents].ID)
      new_weights, new_biases = mutate_weights_biases(cars[parent_index].nn)
      new_car = Car(pos=spawn, rot=-90)
      new_car.nn = PPO(11, 64, 5) 
      new_generation.append(new_car)

    # Update the cars and models with the new generation
    cars = new_generation