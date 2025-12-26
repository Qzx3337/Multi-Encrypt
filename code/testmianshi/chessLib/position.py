import sys

sys.path.append('/tmp/pycharm_project_481/code/testmianshi')

class Position:
    def __init__(self, x: int, y: int) -> object:
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @property
    def x(self) -> int:
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = x

    @property
    def y(self) -> int:
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y

    def to_string(self) -> str:
        return str(self.x) + "," + str(self.y)
