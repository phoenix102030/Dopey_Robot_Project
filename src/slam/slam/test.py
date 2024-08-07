#!/usr/bin/env python3
import numpy as np
import csv
from scipy.spatial import KDTree
from scipy.ndimage import maximum_filter


idk = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
orig_map = np.array(idk)
clean_map = np.zeros((orig_map.shape[0], orig_map.shape[1]))
radius = 2
assert clean_map.shape == orig_map.shape

def clean(clean_map, map):
    for x in range(map.shape[0]):
        for y in range(map.shape[1]):
            if map[x][y] == 1:
                add_pt = 0
                for i in range(x-radius, x+radius+1, 1):
                    for j in range(y-radius, y+radius+1,1): 
                        if i ==x  and j == y:
                            pass
                        else:
                            if 0<= i < map.shape[0] and 0<= j< map.shape[1]:
                                if map[i][j] == 1:
                                    add_pt += 1
                            if add_pt > 1:
                                clean_map[x][y] = 1
    return clean_map

def downsample(arr, window_size):
    result = []
    for i in range(0, len(arr), window_size):
        row_result = []
        for j in range(0, len(arr[0]), window_size):
            window = [arr[x][j:min(j+window_size, len(arr[0]))] for x in range(i, min(i+window_size, len(arr)))]
            if any(num != 0 for sublist in window for num in sublist):
                row_result.append(1)
            else:
                row_result.append(0)
        
        result.append(row_result)
    
    return np.array(result)

def get_obstacles(map):
    obstacle_array = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i][j] ==1:
                obstacle_array.append((i,j, 0))

    return np.array(obstacle_array)

def main():
    result = clean(clean_map, orig_map)
    print(result)
    result = downsample(result, 2)
    print(result)
    obstacles = get_obstacles(result)
    print(obstacles)

if __name__=="__main__":
    main()

