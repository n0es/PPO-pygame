import pygame, sys, math, torch, random, os, shutil
import numpy as np
import pickle
from torch import nn

from raytracer import RayTracer
from pygame.locals import QUIT

def save_model(model, generation, save_path='./saves', file_name_prefix='best_car'):
  os.makedirs(save_path, exist_ok=True)
  file_path = os.path.join(save_path, f'{file_name_prefix}_generation_{generation}.pkl')
  with open(file_path, 'wb') as file:
      pickle.dump(model, file)

  print(f"Model saved at {file_path}")

def load_initial_model(file_name):
  with open(file_name, 'rb') as file:
    initial_model = pickle.load(file)
  return initial_model

def convert_to_float32(module):
  for child in module.children():
    convert_to_float32(child)
  for param in module.parameters():
    param.data = param.data.float()
    param.requires_grad = True

def mutate_model(parent_model, parent_fitness=0.0, mutation_scale=0.5):
  # Determine dynamic mutation rate
  def calculate_mutation_rate(parent_fitness):
    return max(0.1, parent_fitness / 10)

  # Apply mutation to a weight matrix
  def mutate_array(arr, mutation_rate, mutation_scale):
    mutation_mask = np.random.rand(*arr.shape) < mutation_rate
    mutation_values = np.random.uniform(-mutation_scale, mutation_scale, arr.shape)
    return arr + (mutation_mask * mutation_values)

  mutation_rate = calculate_mutation_rate(parent_fitness)

  parent_weights = [p.data.numpy() for p in parent_model.parameters() if p.requires_grad]
  mutated_weights = [mutate_array(w, mutation_rate, mutation_scale) for w in parent_weights]
  mutation_tensors = [torch.tensor(w, requires_grad=True) for w in mutated_weights]

  for i, tensor in enumerate(parent_model.parameters()):
    if tensor.requires_grad:
      tensor.data = mutation_tensors.pop(0)

  return parent_model

def create_new_generation(car_scores, num_best_parents=2, elitism=2, population_size=10):
  sorted_cars_by_score = [car_score[0] for car_score in sorted(car_scores, key=lambda x: x[1], reverse=True)]
  sorted_normalized_scores = np.array([normalized_score for _, normalized_score in sorted(car_scores, key=lambda x: x[1], reverse=True)])
  for i in range(len(sorted_normalized_scores)):
    sorted_normalized_scores[i] += abs(sorted_normalized_scores[-1])

  total_scores = np.sum(sorted_normalized_scores)
  probability_dist = sorted_normalized_scores / total_scores

  best_parents = sorted_cars_by_score[:num_best_parents]

  new_generation = best_parents[:elitism].copy()

  for _ in range(population_size - elitism):
    parent1, parent2 = np.random.choice(sorted_cars_by_score, size=2, p=probability_dist)

    # Crossover parent weights
    child_weights = []
    parent1_weights = [p.data.numpy() for p in parent1.nn.parameters() if p.requires_grad]
    parent2_weights = [p.data.numpy() for p in parent2.nn.parameters() if p.requires_grad]

    for parent1_w, parent2_w in zip(parent1_weights, parent2_weights):
      crossover_mask = np.random.randint(0, 2, parent1_w.shape).astype(bool)
      child_weight = np.where(crossover_mask, parent1_w, parent2_w)
      child_weights.append(child_weight)

    new_car = Car(pos=spawn, rot=-90)
    child_nn = mutate_model(parent1.nn)

    # Replace child NN weights with crossover weights
    for idx, tensor in enumerate(child_nn.parameters()):
      if tensor.requires_grad:
        tensor.data = torch.tensor(child_weights[idx], requires_grad=True)

    new_car.nn = child_nn
    new_generation.append(new_car)

  return new_generation


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
    p1 = np.array(self.p1)
    p2 = np.array(self.p2)
    return np.linalg.norm(p1 - p2)

  def intersects(self, line2):
    x1, y1, x2, y2 = self.points
    x3, y3, x4, y4 = line2.points
    numerator1 = (x1 * y2 - y1 * x2)
    numerator2 = (x3 * y4 - y3 * x4)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
      return False
    px = (numerator1 * (x3 - x4) - (x1 - x2) * numerator2) / denom
    py = (numerator1 * (y3 - y4) - (y1 - y2) * numerator2) / denom
    if (np.min([x1, x2]) <= px <= np.max([x1, x2]) and np.min([y1, y2]) <= py <= np.max([y1, y2]) and
      np.min([x3, x4]) <= px <= np.max([x3, x4]) and np.min([y3, y4]) <= py <= np.max([y3, y4])):
      return True
    return False

class Car: 
  def __init__(self, pos=None, rot=None, ID='0', color='#ff0000', weights=[], biases=[]):
    pos = pos or [300,140]
    rot = rot or -90
    self.dead = False    # [200,75]
    self.pos = pos
    self.rot = rot
    self.ID = ID
    self.color = color
    self.vel = [0,0]
    self.accel = 2
    self.speed = 6
    self.friction = 0.02
    self.width = 6
    self.height = 10
    self.rayTracer = RayTracer(pos, rot, 250, (-fov//2, fov//2), fov//tracers)
    self.best_score = 0
    self.best_weights = None
    self.best_biases = None
    self.initial = {
      'pos': pos,
      'rot': rot,
      'vel': [0,0],
      'accel': 2,
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
    self.nn = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Sigmoid()
        )

    self.score = 0
    self.age = 0.0
    self.time_since_checkpoint = 0.0

  def reset(self):
    self.dead = False
    self.score = 0
    self.age = 0.0
    self.time_since_checkpoint = 0.0
    self.pos = self.initial['pos'].copy()
    self.rot = self.initial['rot']
    self.vel = self.initial['vel'].copy()
  
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
    action_probs = action_probs.squeeze()
    arr = [np.random.choice(2, p=[1-prob, prob]) for prob in action_probs]
    return int(''.join([str(i) for i in arr]), 2)
  

  def cap_velocity(self, velocity, max_speed=1.5):
      magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2)
      if magnitude > max_speed:
          normalized_velocity = [component / magnitude for component in velocity]
          return [component * max_speed for component in normalized_velocity]
      else: return velocity

  def passed_checkpoint(self, checkpoint_lines):
    for i in range(4):
      if self.intersects(checkpoint_lines[self.score]):
        return True
    return False

  def calculate_reward(self):
    checkpoint_reward = 50
    time_penalty = 5
    death_pentalty = 5
    speed_penalty_weight = 5

    speed = math.sqrt(self.vel[0]**2 + self.vel[1]**2)
    min_speed_threshold = 1
    speed_penalty = 0 if speed >= min_speed_threshold else abs(min_speed_threshold - speed) * speed_penalty_weight

    reward = (checkpoint_reward * self.score**2) / (self.age + time_penalty) - (death_pentalty if self.dead else 0) - speed_penalty
    
    return reward
  
  def get_reward_color(self):
    reward = self.calculate_reward()

    def clamp(value, minimum, maximum):
      return max(minimum, min(value, maximum))

    if reward >= 0:
      green_value = clamp(int((reward / 40) * 255), 0, 255)
      color = (255 - green_value, 255, 0)
    else:
      red_value = clamp(int(((abs(reward)) / 40) * 255), 0, 255)
      color = (255, 255 - red_value, 0)

    return color

  
  def render(self, screen):
    self.rayTracer.display(screen,track.walls)
    color = self.get_reward_color()
    pygame.draw.polygon(screen, color, car.vertices())

    # Render car's reward as text
    car_reward = round(self.calculate_reward(), 2)
    reward_text = font.render(str(car_reward), True, (0, 0, 0))
    screen.blit(reward_text, (self.pos[0] - 10, self.pos[1] - 10))

  def update(self, tick):
    if self.dead:
      return
    dt = tick/1000
    self.age += dt
    self.time_since_checkpoint += dt
    self.color = '#ff0000' if self.time_since_checkpoint > 5 else '#00ff00'

    self.rayTracer.pos = self.pos
    self.rayTracer.rot = self.rot

    convert_to_float32(self.nn)
    nn_input = self.rayTracer.distances + self.vel + [self.rot]
    nn_input = torch.tensor(nn_input, dtype=torch.float32, device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))).unsqueeze(0)
    predicted_output = self.nn(nn_input).detach().cpu().numpy()
    action = self.choose_action(predicted_output)

    sin_rot = math.sin(math.radians(self.rot))
    cos_rot = math.cos(math.radians(self.rot))
    
    acceleration = [0, 0]

    if action & 0b10000 > 0:
      acceleration[0] -= sin_rot * self.accel * dt
      acceleration[1] -= cos_rot * self.accel * dt
    if action & 0b01000 > 0:
      acceleration[0] += sin_rot * self.accel * dt
      acceleration[1] += cos_rot * self.accel * dt
    if action & 0b00100 > 0:
      self.rot += 150 * dt
    if action & 0b00010 > 0:
      self.rot -= 150 * dt
    if action & 0b00001 > 0:
      self.vel[0] = self.vel[0] * .9
      self.vel[1] = self.vel[1] * .9

    # Apply friction
    friction_force = [-self.friction * self.vel[0], -self.friction * self.vel[1]]
    self.vel[0] += acceleration[0] + friction_force[0]
    self.vel[1] += acceleration[1] + friction_force[1]

    self.vel = self.cap_velocity(self.vel, self.speed)
    self.pos[0] += self.vel[0]
    self.pos[1] += self.vel[1]
    
    if self.passed_checkpoint(track.checkpoints):
      self.score += 1
      self.time_since_checkpoint = 0.0

pi = 3.141592653589793
score = 0
FPS = 90
multiplier = 10
tracers = 8
fov = 360
spawn = [300,140]
width,height = 800,600

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('vroom!')
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 10)

initial_model_file = None # os.path.join(os.path.dirname(__file__), "../saves/best_car_generation_1000.pkl")
track_file = os.path.join(os.path.dirname(__file__), '../tracks/new.track')
track = Track(track_file)

if initial_model_file:
  initial_model = load_initial_model(initial_model_file)
else:
  initial_model = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Sigmoid()
        )

population_size = 10
cars = [Car() for _ in range(population_size)]
for i, car in enumerate(cars):
  car.nn = initial_model
  if not (i == 0 and initial_model):
    car.nn = mutate_model(car.nn)
car_scores = []

generations = 1000
num_best_parents = 2
elitism = 2

for generation in range(generations):
    print(f"Generation {generation + 1}")
    for car in cars:
      car.pos = spawn
      car.rot = -90
      car.vel = [0,0]
      car.dead = False
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
          car.dead = True
          car.vel = [0,0]

        if all(car.dead for car in cars):
          running = False

      # frame rate monitor
      fps = font.render(str(int(clock.get_fps())), True, pygame.Color('Red'))
      screen.blit(fps, (0, 0))
      # cars monitor
      cars_text = font.render(str(len(cars)), True, pygame.Color('Red'))
      screen.blit(cars_text, (0, 10))

      pygame.display.flip()

    car_scores = []
    for car in cars:
      car_scores.append((car, car.calculate_reward()))

    print(len(car_scores))
    new_generation = create_new_generation(car_scores, num_best_parents, elitism, population_size)

    best_parents = sorted(car_scores, key=lambda x: x[1], reverse=True)[:num_best_parents]

    if (generation + 1) % 25 == 0:
      best_car = best_parents[0]
      save_model(best_car.nn, generation + 1)

    # Update the cars and models with the new generation
    cars = new_generation
    for car in cars:
      car.reset()

    # Update the cars and models with the new generation
    cars = new_generation
    for car in cars:
      car.reset()
