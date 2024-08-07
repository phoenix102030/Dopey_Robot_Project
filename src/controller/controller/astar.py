#!/usr/bin/env python

import numpy as np

from astar import AStar

class astar(AStar):
    def __init__(self, map):
        self.map = map

    def heuristic_cost_estimate(self, n1, n2):
        (x1,y1) = n1    
        (x2, y2) = n2
        return np.sqrt((x1-x2)**2 +(y1-y2)**2)
    
    def distance_between(self, n1, n2):
        (x1,y1) = n1    
        (x2, y2) = n2
        return 1
        return abs(x1-x2)+abs(y1-y2)

    def neighbors(self, node):
        x, y = node
        neighbours = [(x-1,y),(x+1,y),(x,y-1),(x,y+1), (x-1,y-1),(x+1,y+1),(x+1,y-1),(x-1,y+1)]
        to_remove=[]
        for things in neighbours:
            currx, curry = things
            if abs(currx) > 999 or abs(curry)> 999:
                return
            if self.map[int(curry),int(currx)] == -2:
                to_remove.append((things))
            elif self.map[int(curry),int(currx)] == 100:
                to_remove.append((things))
            elif self.map[int(curry), int(currx)] == -128 or self.map[int(curry), int(currx)] == 128:
                to_remove.append((things))
            # elif self.map[int(curry), int(currx)] == -1:
            #     to_remove.append((things))
        for remove in to_remove:
            neighbours.remove(remove)
        return neighbours

    def is_goal_reached(self, current, goal):
        if current ==  goal:
            return True
        else:
            return False
    
def main():
    node = astar()


if __name__ == '__main__':
    main()