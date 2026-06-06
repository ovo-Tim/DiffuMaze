# Multi-goal multi-layer non-intersecting maze map generator
This is a Rust maze map generator that creates multi-goal, multi-layer, non-intersecting maze maps. The results will be saved as int8 Safetensors file.

## Map Format
Let's say we have a g goal, l layer and mxn maze map. The shape of the map is (l, g+1, m, n) where g+1 is the number of channels in the map. Each layer has following channels:
- `0`: The wall channel
- `g`: The checkpoint channel. We don't differentiate between the starting point and the endpoint. You just need to pass through all checkpoints.

Besides the puzzle map, we also provide the solution map. And the shape of the solution map is just (l, m, n). (0 for wall/empty, 1 for path)

## Parameters
- `-w <width>`: The width of the maze map. Default 64
- `-h <height>`: The height of the maze map. Default 64
- `-l <layer>`: The number of layers in the maze map. Default 2
- `-g <goal>`: The number of distinct routes to generate in the maze map. Default 2.
- `-c <checkpoint>`: The number of checkpoints per route. Must be greater than or equal to 2 (e.g., a setting of 2 means just a start point and an endpoint for that route). Total checkpoints in the map will be $g \times c$. Default 2.
- `-n <num>`: The number of maze maps to generate. Default 5.
- `-o <output>`: The output path to save the generated maze maps. Default "maze.safetensors".
- `-t <thread>`: The number of threads to use. Default is cpu core count - 1.
- `-r <image path>`: Render the maze solution as images for human, showing different routes with different colors. Default is not rendering. Default image path is "rendered/"
- `-v <via>`: The number of via you expect a solution route to pass through. Default 1, meaning the algorithm will try it's best to generate maps, which require 1 via in each route to solve. Note that for effeciency, the algorithm will not guarantee to generate maps with exact via number.

## About non-intersecting
No two routes will cross each other in the same layer. However, there is no restriction on the routes in different layers.

# Example
![example1](example1.png)
![example2](example2.png)