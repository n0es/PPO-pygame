import math, pygame, numpy as np


def distance(p1, p2):
  return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def line_intersection(line1, line2):
  x1, y1, x2, y2 = line1
  x3, y3, x4, y4 = line2
  den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
  if den == 0:
    return False
  px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) *
        (x3 * y4 - y3 * x4)) / den
  py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) *
        (x3 * y4 - y3 * x4)) / den
  if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)
      and min(x3, x4) <= px <= max(x3, x4)
      and min(y3, y4) <= py <= max(y3, y4)):
    return True
  return False


class RayTracer:
  def __init__(self, pos, rot, length, angle_range=(-45, 60), angle_step=15):
    self.pos = pos
    self.rot = rot
    self.length = length
    self.angle_range = angle_range
    self.angle_step = angle_step
    self.rays = []
    self.distances = []
    for angle_offset in range(self.angle_range[0], self.angle_range[1],
                              self.angle_step):
      angle = -self.rot + angle_offset
      ray = (self.pos[0] + self.length * math.sin(math.radians(angle)),
             self.pos[1] - self.length * math.cos(math.radians(angle)))
      self.rays.append(ray)
      self.distances.append(self.length)
        


  def display(self, screen, walls):
    self.distances = []
    for ray in self.rays:
      min_intersection = None
      min_dist = float('inf')

      for line in walls:
        intersection = self.ipoint((self.pos[0], self.pos[1], *ray), line.points)
        if intersection:
          dist = distance(self.pos, intersection)
          if dist < min_dist:
            min_dist = dist
            min_intersection = intersection
      if min_intersection:
        pygame.draw.line(
          screen, 'green',
          (self.pos[0], self.pos[1]),
          (min_intersection[0],
           min_intersection[1]))
        self.distances.append(min_dist)
      else:
        pygame.draw.line(
          screen, 'green',
          (self.pos[0], self.pos[1]),
          (ray[0], ray[1]))
        self.distances.append(self.length)

  def check_collisions(self, lines):
    collisions = []
    for ray in self.rays:
      for line in lines:
        if line_intersection((self.pos[0], self.pos[1], *ray), line):
          collisions.append(ray)
          break
    return collisions

  def ipoint(self, line1, line2):
    line1 = np.around(line1).astype(int)
    line2 = np.around(line2).astype(int)
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    den = (x1 - x2) * (y3 - y4) - (x3 - x4) * (y1 - y2)
    if den == 0:
      return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) *
          (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) *
          (x3 * y4 - y3 * x4)) / den
    if (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)
        and min(x3, x4) <= px <= max(x3, x4)
        and min(y3, y4) <= py <= max(y3, y4)):
      return px, py
    return None
