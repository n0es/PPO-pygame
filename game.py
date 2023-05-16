import pygame, sys, math, torch, random
import numpy as np

from neuralnetwork import PPO
from raytracer import RayTracer
from camera import Camera
from collections import deque
from pygame.locals import QUIT

def load_track(file_name):
  walls = []
  checkpoints = []
  current_mode = None

  with open(file_name, 'r') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue

      if line.endswith(':'):
        current_mode = line[:-1]
      else:
        x1, y1, x2, y2 = [int(i) for i in line.split(', ')]
        if current_mode == 'WALLS':
          walls.append((x1, y1, x2, y2))
        elif current_mode == 'CHECKPOINTS':
          checkpoints.append((x1, y1, x2, y2))

  return walls, checkpoints

def choose_action(action_probs):
    return np.random.choice(5, p=action_probs)

def load_track(file_name):
  walls = []
  checkpoints = []
  current_mode = None

  with open(file_name, 'r') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue

      if line.endswith(':'):
        current_mode = line[:-1]
      else:
        x1, y1, x2, y2 = [int(i) for i in line.split(', ')]
        if current_mode == 'WALLS':
          walls.append((x1, y1, x2, y2))
        elif current_mode == 'CHECKPOINTS':
          checkpoints.append((x1, y1, x2, y2))

  return walls, checkpoints

def draw_rect_alpha(surface, color, rect):
  shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
  pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
  surface.blit(shape_surf, rect)

def distance(p1,p2):
  return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

def cap_velocity(velocity, max_speed=1.5):
    magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2)
    if magnitude > max_speed:
        normalized_velocity = [component / magnitude for component in velocity]
        return [component * max_speed for component in normalized_velocity]
    else:
        return velocity

def line_intersection(line1, line2):
  x1, y1, x2, y2 = line1
  x3, y3, x4, y4 = line2
  den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
  if den == 0:
    return False
  px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
  py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
  if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and
    min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
    return True
  return False
  
def detect_collision(car_vertices, track, camera):
  for i in range(len(car_vertices)):
    car_line = (car_vertices[i][0], car_vertices[i][1], car_vertices[i - 1][0], car_vertices[i - 1][1])
    for line in track:
      if line_intersection(car_line, (line[0]-camera.pos[0],line[1]-camera.pos[1],line[2]-camera.pos[0],line[3]-camera.pos[1])):
        return True
  return False

def passed_checkpoint(v, checkpoint_line):
  for i in range(4):
    if line_intersection((v[i][0],v[i][1],v[(i+1)%4][0],v[(i+1)%4][1]), checkpoint_line):
      return True
  return False

def reset_car(pos, rot, vel, layer_sizes):
  global score
  score = 0
  pos[0] = 200
  pos[1] = 55
  rot[0] = -90
  rot[1] = 0
  vel[0] = 0
  vel[1] = 0
  new_nn = PPO(11, 64, 5, device)
  return pos, rot, vel, new_nn

def calculate_reward(score, time_since_checkpoint):
  checkpoint_reward = 100
  time_penalty = -1

  reward = checkpoint_reward * score \
         + time_penalty * time_since_checkpoint
  
  return reward

pi = 3.141592653589793
score = 0
FPS = 30
multiplier = 10
tracers = 8
fov = 360

width,height = 800,600


time_elapsed = time_since_checkpoint = 0

rays = []
accel = .2
speed = 4
keys = []
startPos = pos = [200,55]
rot = [-90, 0]
vel = [0,0]

camera = Camera()
ray_tracer = RayTracer(pos, rot, 250, (-fov//2, fov//2), fov//tracers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn = PPO(11, 64, 5, device)

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('vroom!')
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 10)

walls, checkpoints = load_track('track.txt')


def preprocess_experiences(states, actions, rewards, next_states, dones, gamma=0.99, lambd=0.95):
    values = nn.value_function(torch.tensor(states, dtype=torch.float32).to(device)).cpu().detach().numpy().squeeze()
    next_values = nn.value_function(torch.tensor(next_states, dtype=torch.float32).to(device)).cpu().detach().numpy().squeeze()
    td_errors = rewards + (1 - dones) * gamma * next_values - values
    advantages = []
    gae = 0
    for delta_t in reversed(td_errors):
        gae = delta_t + gamma * lambd * gae
        advantages.insert(0, gae)
    returns = advantages + values
    return np.array(states), np.array(actions), np.array(advantages), np.array(returns)

def collect_experiences():
    global pos, rot, vel, score, time_elapsed, time_since_checkpoint, ray_tracer, nn
    experiences = []
    while len(experiences) < EXPERIENCES_PER_UPDATE:
        
        pygame.draw.rect(screen, (255,255,255), pygame.Rect(0, 0, width, height))
        tick = clock.tick(120)
        dt = tick / (1000/(FPS*multiplier))
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
        track = walls
        ray_tracer.display(screen,track,camera)

        for i in keys:
          match i:
            case pygame.K_w:
              vel[0] -= math.sin(math.radians(rot[0]))*accel*dt
              vel[1] -= math.cos(math.radians(rot[0]))*accel*dt
            case pygame.K_s:
              vel[0] += math.sin(math.radians(rot[0]))*accel*dt
              vel[1] += math.cos(math.radians(rot[0]))*accel*dt
            case pygame.K_a:
              rot[0] += .3*dt
            case pygame.K_d:
              rot[0] -= .3*dt
            case pygame.K_SPACE:
              vel[0] = vel[0]*.9
              vel[1] = vel[1]*.9

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

        sin = math.sin(math.radians(rot[0]))
        cos = math.cos(math.radians(rot[0]))
        cheight = 10
        cwidth = 6
        vertices = [
          [pos[0]-camera.pos[0] + sin*cheight/2 - cos*cwidth/2,
          pos[1]-camera.pos[1] + cos*cheight/2 + sin*cwidth/2],
          [pos[0]-camera.pos[0] - sin*cheight/2 - cos*cwidth/2,
          pos[1]-camera.pos[1] - cos*cheight/2 + sin*cwidth/2],
          [pos[0]-camera.pos[0] - sin*cheight/2 + cos*cwidth/2,
          pos[1]-camera.pos[1] - cos*cheight/2 - sin*cwidth/2],
          [pos[0]-camera.pos[0] + sin*cheight/2 + cos*cwidth/2,
          pos[1]-camera.pos[1] + cos*cheight/2 - sin*cwidth/2],
        ]

        # lag the camera behind the car
        camera.pos[0] = (((pos[0]-width/2) - camera.pos[0]) / 2) + camera.pos[0]
        camera.pos[1] = (((pos[1]-height/2) - camera.pos[1]) / 2) + camera.pos[1]
        camera.rot = ((rot[0]-camera.rot)/1.3)+camera.rot

        pygame.draw.polygon(screen, 'red', vertices)
        # for i in range(len(vertices)):
        #   vertices[i][0]+=camera.pos[0]
        #   vertices[i][1]+=camera.pos[1]

        coords = font.render(f'[{pos[0]//1},{pos[1]//1}], {rot[0]}*', True, (255, 0, 0))
        screen.blit(coords, (5, 5))
        # draw track lines
        for line in track:
          pygame.draw.line(screen, 'red', (line[0] - camera.pos[0], line[1] - camera.pos[1]),
                            (line[2] - camera.pos[0], line[3] - camera.pos[1]), 2)
        # draw checkpoint lines
        for i,line in enumerate(checkpoints):
          pygame.draw.line(screen,'blue', (line[0] - camera.pos[0], line[1] - camera.pos[1]),
                            (line[2] - camera.pos[0], line[3] - camera.pos[1]), 2)
          # draw checkpoint number
          num = font.render(str(i), True, (0, 0, 0))
          screen.blit(num, ((line[0] + line[2]) / 2 - camera.pos[0], (line[1] + line[3]) / 2 - camera.pos[1]))
        
          # Added check for passing the checkpoint and increase the score
        if len(checkpoints) > 0:
          if detect_collision(vertices, [checkpoints[score % len(checkpoints)]], camera):
            score += 1
            time_since_checkpoint = 0

          # Check for collision and reset the car if there's a collision
        if detect_collision(vertices, track, camera) or time_since_checkpoint >= 15:
          time_elapsed = time_since_checkpoint = 0 
          pos, rot, vel, nn = reset_car(pos, rot, vel, [11, 9, 5])
          ray_tracer = RayTracer(pos, rot, 250, (-fov//2, fov//2), fov//tracers)

          reward = calculate_reward(score, time_elapsed)
          done = True
          experiences.append((nn_input, action, reward, nn_input, float(done)))
          continue

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
        velocity_vector = (pos[0]-camera.pos[0] + vel[0] * velocity_scale, pos[1]-camera.pos[1] + vel[1] * velocity_scale)
        pygame.draw.line(screen, (0, 0, 255), (pos[0]-camera.pos[0],pos[1]-camera.pos[1]), velocity_vector, 2)
          
        pygame.display.update()
        if collect_experiences_flag:
          done = detect_collision(vertices, track, camera) or time_since_checkpoint >= 15
          reward = calculate_reward(score, time_since_checkpoint)
          nn_input_next = nn_input[:]

          experiences.append((nn_input, action, reward, nn_input_next, float(done)))
        
        if done:
            pos, rot, vel, nn = reset_car(pos, rot, vel, [11, 64, 5])
            ray_tracer = RayTracer(pos, rot, 250, (-fov // 2, fov // 2), fov // tracers)
            time_elapsed = time_since_checkpoint = 0
            score = 0

    return experiences

EXPERIENCES_PER_UPDATE = 3072
UPDATES_PER_ITERATION = 10
ITERATIONS = 50
collect_experiences_flag = False

iteration = 0
while iteration < ITERATIONS:
    collect_experiences_flag = True
    step_rewards = 0

    experiences = collect_experiences()
    states, actions, rewards, next_states, dones = zip(*experiences)
    states, actions, advantages, returns = preprocess_experiences(states, actions, rewards, next_states, dones)

    for _ in range(UPDATES_PER_ITERATION):
        log_probs_old = nn.pol_eval(torch.tensor(states, dtype=torch.float32).to(device)).gather(
            1, torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)).log().cpu().detach().numpy()
        nn.update(states, actions, log_probs_old.squeeze(), returns, advantages)

    collect_experiences_flag = False
    iteration += 1