
from math import *

class Point:
    x = 0
    y = 0
    z = 0

    def set_location(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance_from_origin(self):
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def distance(self, other):

        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z

        return sqrt(dx * dx + dy * dy + dz * dz)



