use clap::Parser;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::PathBuf;

// ==================== CLI Arguments ====================

#[derive(Parser, Debug)]
#[command(name = "maze-generator", disable_help_flag = true)]
struct Args {
    #[arg(short = 'w', default_value = "64")]
    width: usize,

    #[arg(short = 'h', default_value = "64")]
    height: usize,

    #[arg(short = 'l', default_value = "2")]
    layers: usize,

    #[arg(short = 'g', default_value = "2")]
    goals: usize,

    #[arg(short = 'c', default_value = "2")]
    checkpoints: usize,

    #[arg(short = 'n', default_value = "5")]
    num: usize,

    #[arg(short = 'o', default_value = "maze.safetensors")]
    output: String,

    #[arg(short = 't')]
    threads: Option<usize>,

    #[arg(short = 'r')]
    render: Option<String>,

    #[arg(short = 'v', default_value = "1")]
    via: usize,

    #[arg(long = "help", action = clap::ArgAction::SetTrue)]
    help_flag: bool,
}

// ==================== Data Structures ====================

struct MazeMap {
    puzzle: Vec<i8>,
    solution: Vec<i8>,
}

#[derive(Clone)]
struct LayerRouteData {
    route_owner: Vec<u8>,
    checkpoints: Vec<Vec<(usize, usize)>>,
    vias: Vec<Vec<(usize, usize)>>,
}

// ==================== Maze Grid ====================

fn maze_to_grid(mi: usize, mj: usize) -> (usize, usize) {
    (2 * mi + 1, 2 * mj + 1)
}

#[allow(dead_code)]
fn grid_to_maze(y: usize, x: usize) -> (usize, usize) {
    ((y - 1) / 2, (x - 1) / 2)
}

// ==================== Region Partitioning ====================

fn partition_regions(
    rng: &mut StdRng,
    maze_w: usize,
    maze_h: usize,
    goals: usize,
) -> Vec<HashSet<(usize, usize)>> {
    if goals == 0 {
        return vec![];
    }

    let all_cells: Vec<(usize, usize)> = (0..maze_h)
        .flat_map(|mi| (0..maze_w).map(move |mj| (mi, mj)))
        .collect();

    let mut seeds: Vec<(usize, usize)> = Vec::new();

    seeds.push(all_cells[rng.gen_range(0..all_cells.len())]);

    for _ in 1..goals {
        let mut best = all_cells[0];
        let mut best_dist = 0i64;

        for &cell in &all_cells {
            let min_dist = seeds
                .iter()
                .map(|&s| {
                    let dy = cell.0 as i64 - s.0 as i64;
                    let dx = cell.1 as i64 - s.1 as i64;
                    dy * dy + dx * dx
                })
                .min()
                .unwrap_or(0);

            if min_dist > best_dist {
                best_dist = min_dist;
                best = cell;
            }
        }

        let candidates: Vec<(usize, usize)> = all_cells
            .iter()
            .filter(|&&cell| {
                let min_dist = seeds
                    .iter()
                    .map(|&s| {
                        let dy = cell.0 as i64 - s.0 as i64;
                        let dx = cell.1 as i64 - s.1 as i64;
                        dy * dy + dx * dx
                    })
                    .min()
                    .unwrap_or(0);
                min_dist >= best_dist * 3 / 4
            })
            .copied()
            .collect();

        let next = if candidates.is_empty() {
            best
        } else {
            candidates[rng.gen_range(0..candidates.len())]
        };
        seeds.push(next);
    }

    let mut regions: Vec<HashSet<(usize, usize)>> = vec![HashSet::new(); goals];
    let mut claimed = vec![vec![0usize; maze_w]; maze_h];
    let mut queue: VecDeque<(usize, (usize, usize))> = VecDeque::new();

    for (i, &seed) in seeds.iter().enumerate() {
        regions[i].insert(seed);
        claimed[seed.0][seed.1] = i + 1;
        queue.push_back((i, seed));
    }

    while let Some((region_idx, (mi, mj))) = queue.pop_front() {
        for (dy, dx) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let ni = mi as i32 + dy;
            let nj = mj as i32 + dx;
            if ni >= 0 && nj >= 0 && (ni as usize) < maze_h && (nj as usize) < maze_w {
                let (ni, nj) = (ni as usize, nj as usize);
                if claimed[ni][nj] == 0 {
                    claimed[ni][nj] = region_idx + 1;
                    regions[region_idx].insert((ni, nj));
                    queue.push_back((region_idx, (ni, nj)));
                }
            }
        }
    }

    regions
}

// ==================== DFS Maze Generation ====================

fn generate_maze_dfs(
    rng: &mut StdRng,
    region: &HashSet<(usize, usize)>,
    maze_w: usize,
    maze_h: usize,
) -> HashSet<((usize, usize), (usize, usize))> {
    let mut edges = HashSet::new();
    let mut visited = HashSet::new();

    if region.is_empty() {
        return edges;
    }

    let start = *region.iter().choose(rng).unwrap();
    visited.insert(start);

    let mut stack = vec![start];

    while !stack.is_empty() {
        let (mi, mj) = *stack.last().unwrap();

        let mut neighbors: Vec<(usize, usize)> = Vec::new();
        for (dy, dx) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let ni = mi as i32 + dy;
            let nj = mj as i32 + dx;
            if ni >= 0 && nj >= 0 && (ni as usize) < maze_h && (nj as usize) < maze_w {
                let n = (ni as usize, nj as usize);
                if region.contains(&n) && !visited.contains(&n) {
                    neighbors.push(n);
                }
            }
        }

        if neighbors.is_empty() {
            stack.pop();
        } else {
            let next = neighbors[rng.gen_range(0..neighbors.len())];
            visited.insert(next);

            let edge = if (mi, mj) < next {
                ((mi, mj), next)
            } else {
                (next, (mi, mj))
            };
            edges.insert(edge);

            stack.push(next);
        }
    }

    edges
}

// ==================== BFS Path Finding ====================

fn find_path_in_maze(
    start: (usize, usize),
    end: (usize, usize),
    edges: &HashSet<((usize, usize), (usize, usize))>,
    region: &HashSet<(usize, usize)>,
) -> Vec<(usize, usize)> {
    let mut adj: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for &(a, b) in edges {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }

    for &cell in region {
        adj.entry(cell).or_default();
    }

    let mut visited = HashSet::new();
    let mut parent: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut queue = VecDeque::new();

    visited.insert(start);
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        if current == end {
            break;
        }

        if let Some(neighbors) = adj.get(&current) {
            for &next in neighbors {
                if !visited.contains(&next) {
                    visited.insert(next);
                    parent.insert(next, current);
                    queue.push_back(next);
                }
            }
        }
    }

    let mut path = Vec::new();
    let mut current = end;
    while current != start {
        path.push(current);
        match parent.get(&current) {
            Some(&p) => current = p,
            None => return vec![start, end],
        }
    }
    path.push(start);
    path.reverse();
    path
}

// ==================== Checkpoint Selection ====================

fn pick_checkpoints_in_region(
    rng: &mut StdRng,
    region: &HashSet<(usize, usize)>,
    count: usize,
    exclude: &HashSet<(usize, usize)>,
) -> Vec<(usize, usize)> {
    let available: Vec<(usize, usize)> = region
        .iter()
        .copied()
        .filter(|c| !exclude.contains(c))
        .collect();

    if available.is_empty() {
        let mut fallback: Vec<_> = region.iter().copied().collect();
        fallback.shuffle(rng);
        fallback.truncate(count);
        return fallback;
    }

    if available.len() <= count {
        let mut cps: Vec<_> = available;
        cps.shuffle(rng);
        cps.truncate(count);
        return cps;
    }

    let mut chosen: Vec<(usize, usize)> = Vec::new();
    chosen.push(available[rng.gen_range(0..available.len())]);

    while chosen.len() < count {
        let candidates: Vec<((usize, usize), i64)> = available
            .iter()
            .filter(|&&c| !chosen.contains(&c))
            .map(|&c| {
                let min_dist = chosen
                    .iter()
                    .map(|&ch| {
                        let dy = c.0 as i64 - ch.0 as i64;
                        let dx = c.1 as i64 - ch.1 as i64;
                        dy * dy + dx * dx
                    })
                    .min()
                    .unwrap_or(0);
                (c, min_dist)
            })
            .collect();

        if candidates.is_empty() {
            break;
        }

        let mut candidates = candidates;
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        let top_n = (candidates.len().min(5)).max(1);
        let pick = candidates[rng.gen_range(0..top_n)].0;
        chosen.push(pick);
    }

    chosen
}

// ==================== Full Map Generation ====================

fn generate_map(
    seed: u64,
    width: usize,
    height: usize,
    layers: usize,
    goals: usize,
    checkpoints: usize,
    via_count: usize,
) -> (MazeMap, Vec<LayerRouteData>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let channel_size = height * width;
    let maze_w = width / 2;
    let maze_h = height / 2;

    if maze_w == 0 || maze_h == 0 || maze_w * maze_h < goals * checkpoints {
        let puzzle = vec![1i8; layers * (goals + 1) * channel_size];
        let solution = vec![0i8; layers * channel_size];
        let route_data = (0..layers)
            .map(|_| LayerRouteData {
                route_owner: vec![goals as u8; channel_size],
                checkpoints: (0..goals).map(|_| vec![]).collect(),
                vias: (0..goals).map(|_| vec![]).collect(),
            })
            .collect();
        return (MazeMap { puzzle, solution }, route_data);
    }

    // Step 1: Global region partitioning (shared across ALL layers)
    let regions = partition_regions(&mut rng, maze_w, maze_h, goals);

    // Step 2: Generate per-layer perfect mazes for each region
    let mut layer_edges: Vec<Vec<HashSet<((usize, usize), (usize, usize))>>> = Vec::new();
    let mut rng_for_layers = rng.clone();
    for _ in 0..layers {
        let mut edges_per_region = Vec::new();
        for r in 0..goals {
            let edges = generate_maze_dfs(&mut rng_for_layers, &regions[r], maze_w, maze_h);
            edges_per_region.push(edges);
        }
        layer_edges.push(edges_per_region);
    }
    rng = rng_for_layers;

    // Step 3: Place vias for each route
    // Each route gets `via_count` vias, each via is a maze cell position
    // that exists in ALL layers (the cell is carved as passage in all layers)
    // When via_count > 0, each route will have its checkpoints distributed
    // across layers, requiring layer transitions through vias to solve.
    let mut route_vias: Vec<Vec<(usize, usize)>> = vec![Vec::new(); goals];

    if via_count > 0 && layers > 1 {
        for r in 0..goals {
            let region_cells: Vec<(usize, usize)> = regions[r].iter().copied().collect();
            // Pick via positions within this route's region
            let num_vias = via_count.min(region_cells.len());
            let mut chosen_vias = Vec::new();
            let mut used = HashSet::new();

            for _ in 0..num_vias {
                let candidates: Vec<(usize, usize)> = region_cells
                    .iter()
                    .filter(|&&c| !used.contains(&c))
                    .copied()
                    .collect();

                if candidates.is_empty() {
                    break;
                }

                if chosen_vias.is_empty() {
                    let pick = candidates[rng.gen_range(0..candidates.len())];
                    chosen_vias.push(pick);
                    used.insert(pick);
                } else {
                    // Farthest-point sampling from existing vias
                    let mut scored: Vec<((usize, usize), i64)> = candidates
                        .iter()
                        .map(|&c| {
                            let min_dist = chosen_vias
                                .iter()
                                .map(|&v| {
                                    let dy = c.0 as i64 - v.0 as i64;
                                    let dx = c.1 as i64 - v.1 as i64;
                                    dy * dy + dx * dx
                                })
                                .min()
                                .unwrap_or(0);
                            (c, min_dist)
                        })
                        .collect();
                    scored.sort_by(|a, b| b.1.cmp(&a.1));
                    let top_n = scored.len().min(3).max(1);
                    let pick = scored[rng.gen_range(0..top_n)].0;
                    chosen_vias.push(pick);
                    used.insert(pick);
                }
            }
            route_vias[r] = chosen_vias;
        }
    }

    // Step 4: Assign checkpoints to layers for each route
    // Each route has `checkpoints` checkpoints. Distribute them across layers
    // so that when via_count > 0, the route must cross layers to visit all checkpoints.
    // Layout: start on layer 0, via to layer 1, ..., end on layer determined by route length
    let mut route_layer_checkpoints: Vec<Vec<Vec<(usize, usize)>>> =
        vec![vec![Vec::new(); layers]; goals];

    for r in 0..goals {
        // Determine which layers each checkpoint goes on
        // Total segments = checkpoints - 1
        // With v vias, we need v+1 segments across v+1 layer visits
        // Distribute checkpoints: first checkpoint on layer 0, then transition
        // through vias to other layers
        let total_segments = if via_count > 0 && layers > 1 {
            (via_count + 1).min(checkpoints)
        } else {
            1
        };
        let checkpoints_per_segment = if total_segments >= checkpoints {
            1
        } else {
            checkpoints / total_segments
        };

        // Assign checkpoints to layer sequence
        // When via_count > 0: layer sequence is 0,1,0,1,... (or 0,1,2,... for more layers)
        let mut cp_idx = 0;
        let mut all_cps = Vec::new();

        // Exclude via positions from checkpoint candidates
        let via_set: HashSet<(usize, usize)> = route_vias[r].iter().copied().collect();

        for seg in 0..total_segments {
            let layer = if layers > 1 {
                seg % layers
            } else {
                0
            };

            let num_cps_in_seg = if seg == total_segments - 1 {
                checkpoints.saturating_sub(cp_idx)
            } else {
                checkpoints_per_segment.max(1)
            };

            let cps = pick_checkpoints_in_region(
                &mut rng,
                &regions[r],
                num_cps_in_seg,
                &via_set,
            );

            for &(mi, mj) in &cps {
                let (y, x) = maze_to_grid(mi, mj);
                route_layer_checkpoints[r][layer].push((y, x));
                all_cps.push((layer, mi, mj));
            }
            cp_idx += num_cps_in_seg;
        }

        // If we didn't place enough checkpoints, place remaining on layer 0
        while all_cps.len() < checkpoints {
            let cps = pick_checkpoints_in_region(
                &mut rng,
                &regions[r],
                1,
                &via_set,
            );
            if let Some(&(mi, mj)) = cps.first() {
                let (y, x) = maze_to_grid(mi, mj);
                route_layer_checkpoints[r][0].push((y, x));
                all_cps.push((0, mi, mj));
            } else {
                break;
            }
        }
    }

    // Step 5: Build the multi-layer route (solution path)
    // For each route, construct the full path:
    //   - Within each layer, BFS path from checkpoint/via to next via/checkpoint
    //   - At vias, transition between layers
    //   - The solution shows which cells are part of the route in each layer

    let mut layer_solution_cells: Vec<Vec<HashSet<(usize, usize)>>> =
        vec![vec![HashSet::new(); goals]; layers];
    let mut layer_grid: Vec<Vec<Vec<bool>>> =
        vec![vec![vec![false; width]; height]; layers];
    let mut layer_cell_owner: Vec<Vec<Vec<usize>>> =
        vec![vec![vec![0; width]; height]; layers];

    // First, carve all maze passages into the grids
    for l in 0..layers {
        for r in 0..goals {
            let edges = &layer_edges[l][r];

            for &(a, b) in edges {
                let (y1, x1) = maze_to_grid(a.0, a.1);
                let (y2, x2) = maze_to_grid(b.0, b.1);
                let wy = (y1 + y2) / 2;
                let wx = (x1 + x2) / 2;

                layer_grid[l][y1][x1] = true;
                layer_grid[l][y2][x2] = true;
                layer_grid[l][wy][wx] = true;

                layer_cell_owner[l][y1][x1] = r;
                layer_cell_owner[l][y2][x2] = r;
                layer_cell_owner[l][wy][wx] = r;
            }

            for &(mi, mj) in &regions[r] {
                let (y, x) = maze_to_grid(mi, mj);
                layer_grid[l][y][x] = true;
                layer_cell_owner[l][y][x] = r;
            }
        }
    }

    // Carve via positions in ALL layers (vias connect layers)
    for r in 0..goals {
        for &(mi, mj) in &route_vias[r] {
            let (y, x) = maze_to_grid(mi, mj);
            for l in 0..layers {
                layer_grid[l][y][x] = true;
                layer_cell_owner[l][y][x] = r;
                // Also ensure via cell neighborhood connectivity to the region
                // The via position must connect to at least one neighbor in the maze
                // Since the via is within the region, it's already carved as a cell
                // We also need to make sure there's a path FROM the via TO the rest
                // of the maze in each layer. The DFS maze already connects all cells
                // in the region, but the via is a fixed cell, so it should be connected.
                // However, we need to ensure the walls around the via connect it to
                // the maze in each layer. Let's add a connection from the via to its
                // nearest neighbor in the region for each layer.
            }
        }
    }

    // Ensure via connectivity: for each via, ensure it's connected to the rest of
    // the region's maze in each layer by carving a wall between the via cell and
    // its nearest region neighbor.
    for r in 0..goals {
        for &(mi, mj) in &route_vias[r] {
            for l in 0..layers {
                let edges = &layer_edges[l][r];
                // Check if via cell is already connected (has an edge to a neighbor)
                let connected = edges.iter().any(|&(a, b)| {
                    a == (mi, mj) || b == (mi, mj)
                });

                if !connected && regions[r].contains(&(mi, mj)) {
                    // The via cell might be isolated in this layer.
                    // Find a neighboring cell in the region that is connected
                    // and carve the wall between them.
                    let (vy, vx) = maze_to_grid(mi, mj);
                    for (dy, dx) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                        let ni = mi as i32 + dy;
                        let nj = mj as i32 + dx;
                        if ni >= 0 && nj >= 0 && (ni as usize) < maze_h && (nj as usize) < maze_w {
                            let n = (ni as usize, nj as usize);
                            if regions[r].contains(&n) {
                                // Carve wall between (mi,mj) and (ni,nj)
                                let (ny, nx) = maze_to_grid(n.0, n.1);
                                let wy = (vy + ny) / 2;
                                let wx = (vx + nx) / 2;
                                layer_grid[l][wy][wx] = true;
                                layer_cell_owner[l][wy][wx] = r;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 6: Construct the full route path for each route
    // A route visits: cp0_layer -> via0 -> cp1_layer -> via1 -> ... -> cpN_layer
    // Build the ordered sequence of waypoints (checkpoints + vias) with their layers
    for r in 0..goals {
        // Build the waypoint sequence for this route
        // Sequence: checkpoint on layer 0, via[0], checkpoint on next layer, via[1], ...
        // depending on via_count

        // Collect all waypoints in order with their layers
        let mut waypoints: Vec<(usize, (usize, usize))> = Vec::new(); // (layer, (mi, mj))

        // Distribute checkpoints across layers with vias in between
        // Simple approach: alternate layers for checkpoints, insert vias between transitions
        let cps_per_layer = &route_layer_checkpoints[r];

        // Get all checkpoints as (layer, grid_y, grid_x, maze_mi, maze_mj)
        let mut all_cp_ordered: Vec<(usize, usize, usize)> = Vec::new(); // (layer, mi, mj)
        for l in 0..layers {
            for &(gy, gx) in &cps_per_layer[l] {
                let (mi, mj) = grid_to_maze(gy, gx);
                all_cp_ordered.push((l, mi, mj));
            }
        }

        if all_cp_ordered.is_empty() {
            continue;
        }

        // Build waypoints list: cps interspersed with vias for layer transitions
        // Strategy: insert a via waypoint between every consecutive pair of
        // checkpoints that are on different layers
        waypoints.push((all_cp_ordered[0].0, (all_cp_ordered[0].1, all_cp_ordered[0].2)));

        for i in 1..all_cp_ordered.len() {
            let prev_layer = all_cp_ordered[i - 1].0;
            let curr_layer = all_cp_ordered[i].0;

            if prev_layer != curr_layer && !route_vias[r].is_empty() {
                // Need a via to transition layers
                // Use a via in the same region - pick one that's reachable
                let via_idx = (i - 1) % route_vias[r].len();
                waypoints.push((prev_layer, route_vias[r][via_idx]));
                waypoints.push((curr_layer, route_vias[r][via_idx]));
            }

            waypoints.push((curr_layer, (all_cp_ordered[i].1, all_cp_ordered[i].2)));
        }

        // Now find paths between consecutive waypoints within the same layer
        for w in 0..waypoints.len().saturating_sub(1) {
            let (layer_a, cell_a) = waypoints[w];
            let (layer_b, cell_b) = waypoints[w + 1];

            if layer_a != layer_b {
                // This shouldn't happen - vias handle layer transitions
                continue;
            }

            let layer = layer_a;
            let edges = &layer_edges[layer][r];

            let path = find_path_in_maze(cell_a, cell_b, edges, &regions[r]);

            for &(mi, mj) in &path {
                let (y, x) = maze_to_grid(mi, mj);
                layer_solution_cells[layer][r].insert((y, x));
            }
            // Also add wall cells between consecutive path cells
            for j in 0..path.len().saturating_sub(1) {
                let (mi1, mj1) = path[j];
                let (mi2, mj2) = path[j + 1];
                let (y1, x1) = maze_to_grid(mi1, mj1);
                let (y2, x2) = maze_to_grid(mi2, mj2);
                let wy = (y1 + y2) / 2;
                let wx = (x1 + x2) / 2;
                layer_solution_cells[layer][r].insert((wy, wx));
            }
        }
    }

    // Step 7: Build output arrays
    let mut puzzle = vec![0i8; layers * (goals + 1) * channel_size];
    let mut solution = vec![0i8; layers * channel_size];

    let mut result_route_data = Vec::with_capacity(layers);

    for l in 0..layers {
        let layer_offset = l * (goals + 1) * channel_size;
        let sol_offset = l * channel_size;

        // Channel 0: walls
        for y in 0..height {
            for x in 0..width {
                if !layer_grid[l][y][x] {
                    puzzle[layer_offset + y * width + x] = 1;
                }
            }
        }

        // Channels 1..g+1: checkpoints
        for r in 0..goals {
            let cp_offset = layer_offset + (r + 1) * channel_size;
            for &(cy, cx) in &route_layer_checkpoints[r][l] {
                puzzle[cp_offset + cy * width + cx] = 1;
            }
            // Mark via positions as checkpoints in the checkpoint channel
            for &(mi, mj) in &route_vias[r] {
                let (vy, vx) = maze_to_grid(mi, mj);
                if layer_grid[l][vy][vx] {
                    puzzle[cp_offset + vy * width + vx] = 1;
                }
            }
        }

        // Solution: mark path cells
        for r in 0..goals {
            for &(y, x) in &layer_solution_cells[l][r] {
                solution[sol_offset + y * width + x] = 1;
            }
        }

        // Build route owner
        let mut route_owner = vec![goals as u8; channel_size];
        for y in 0..height {
            for x in 0..width {
                if layer_grid[l][y][x] {
                    route_owner[y * width + x] = layer_cell_owner[l][y][x] as u8;
                }
            }
        }

        // Build checkpoints per route for this layer
        let mut layer_checkpoints: Vec<Vec<(usize, usize)>> = Vec::new();
        for r in 0..goals {
            layer_checkpoints.push(route_layer_checkpoints[r][l].clone());
        }

        // Build via list per route for this layer (in grid coordinates)
        let mut layer_vias: Vec<Vec<(usize, usize)>> = vec![Vec::new(); goals];
        for r in 0..goals {
            for &(mi, mj) in &route_vias[r] {
                let (y, x) = maze_to_grid(mi, mj);
                layer_vias[r].push((y, x));
            }
        }

        result_route_data.push(LayerRouteData {
            route_owner,
            checkpoints: layer_checkpoints,
            vias: layer_vias,
        });
    }

    (MazeMap { puzzle, solution }, result_route_data)
}

// ==================== Safetensors Output ====================

fn save_maps(
    maps: &[MazeMap],
    width: usize,
    height: usize,
    layers: usize,
    goals: usize,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = maps.len();
    let channel_size = height * width;

    let total_puzzle_size = n * layers * (goals + 1) * channel_size;
    let total_solution_size = n * layers * channel_size;

    let mut puzzle_data = vec![0i8; total_puzzle_size];
    let mut solution_data = vec![0i8; total_solution_size];

    for (i, map) in maps.iter().enumerate() {
        let puzzle_start = i * map.puzzle.len();
        puzzle_data[puzzle_start..puzzle_start + map.puzzle.len()].copy_from_slice(&map.puzzle);

        let sol_start = i * map.solution.len();
        solution_data[sol_start..sol_start + map.solution.len()]
            .copy_from_slice(&map.solution);
    }

    let puzzle_shape = vec![n, layers, goals + 1, height, width];
    let solution_shape = vec![n, layers, height, width];

    let puzzle_bytes: &[u8] = bytemuck::cast_slice(&puzzle_data);
    let solution_bytes: &[u8] = bytemuck::cast_slice(&solution_data);

    use safetensors::tensor::{Dtype, TensorView};

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert(
        "puzzle".to_string(),
        TensorView::new(Dtype::I8, puzzle_shape, puzzle_bytes)
            .map_err(|e| format!("{e:?}"))?,
    );
    tensors.insert(
        "solution".to_string(),
        TensorView::new(Dtype::I8, solution_shape, solution_bytes)
            .map_err(|e| format!("{e:?}"))?,
    );

    safetensors::serialize_to_file(&tensors, &None, &PathBuf::from(path))
        .map_err(|e| format!("{e:?}"))?;

    Ok(())
}

fn validate_map(
    map: &MazeMap,
    width: usize,
    height: usize,
    layers: usize,
    goals: usize,
    checkpoints: usize,
    via_count: usize,
) -> Vec<String> {
    let channel_size = height * width;
    let mut errors = Vec::new();

    for l in 0..layers {
        let layer_offset = l * (goals + 1) * channel_size;
        let sol_offset = l * channel_size;

        // Check wall/solution overlap
        let mut wall_sol_overlap = 0usize;
        for y in 0..height {
            for x in 0..width {
                let is_wall = map.puzzle[layer_offset + y * width + x] == 1;
                let is_sol = map.solution[sol_offset + y * width + x] == 1;
                if is_wall && is_sol {
                    wall_sol_overlap += 1;
                }
            }
        }
        if wall_sol_overlap > 0 {
            errors.push(format!("Layer {l}: {wall_sol_overlap} wall/solution overlaps"));
        }

        // Check checkpoint count per route (including vias)
        for r in 0..goals {
            let cp_offset = layer_offset + (r + 1) * channel_size;
            let cp_count = (0..channel_size).filter(|&i| map.puzzle[cp_offset + i] == 1).count();
            // Each route should have at least 1 checkpoint on some layer
            // Total checkpoints per route across all layers = checkpoints
            // Plus via positions
        }
    }

    // Check total checkpoints per route across all layers
    for r in 0..goals {
        let mut total_cp = 0usize;
        for l in 0..layers {
            let cp_offset = l * (goals + 1) * channel_size + (r + 1) * channel_size;
            total_cp += (0..channel_size).filter(|&i| map.puzzle[cp_offset + i] == 1).count();
        }
        // Should have at least `checkpoints` + via_count positions marked
        if total_cp < checkpoints {
            errors.push(format!("Route {r}: only {total_cp} checkpoint cells (expected >= {checkpoints})"));
        }
    }

    // Check via positions are passable in all layers
    // (We can't easily check this from the puzzle data alone without route_data)
    // So we check that solution paths exist
    for l in 0..layers {
        let sol_offset = l * channel_size;
        let sol_cells = (0..channel_size).filter(|&i| map.solution[sol_offset + i] == 1).count();
        if via_count > 0 && layers > 1 && sol_cells == 0 {
            errors.push(format!("Layer {l}: no solution path cells"));
        }
    }

    errors
}

// ==================== Image Rendering ====================

const ROUTE_COLORS: [[u8; 3]; 10] = [
    [220, 50, 50],
    [50, 180, 50],
    [50, 80, 220],
    [220, 180, 30],
    [180, 50, 180],
    [50, 180, 180],
    [200, 120, 50],
    [120, 50, 200],
    [50, 200, 100],
    [200, 50, 120],
];

fn render_map(
    map: &MazeMap,
    route_data: &[LayerRouteData],
    width: usize,
    height: usize,
    layers: usize,
    goals: usize,
    output_dir: &str,
    map_idx: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use image::{Rgb, RgbImage};

    let channel_size = height * width;
    fs::create_dir_all(output_dir)?;

    for l in 0..layers {
        let mut img = RgbImage::new(width as u32, height as u32);

        let sol_offset = l * channel_size;
        let puzzle_layer_offset = l * (goals + 1) * channel_size;
        let rd = &route_data[l];

        for y in 0..height {
            for x in 0..width {
                if map.puzzle[puzzle_layer_offset + y * width + x] == 1 {
                    img.put_pixel(x as u32, y as u32, Rgb([0, 0, 0]));
                } else {
                    let owner = rd.route_owner[y * width + x] as usize;
                    if owner < goals && map.solution[sol_offset + y * width + x] == 1 {
                        let color = ROUTE_COLORS[owner % ROUTE_COLORS.len()];
                        img.put_pixel(x as u32, y as u32, Rgb(color));
                    } else {
                        img.put_pixel(x as u32, y as u32, Rgb([128, 128, 128]));
                    }
                }
            }
        }

        // Draw checkpoints: start=white, end=white, intermediate=white
        for r in 0..goals {
            for &(cy, cx) in &rd.checkpoints[r] {
                img.put_pixel(cx as u32, cy as u32, Rgb([255, 255, 255]));
            }
        }

        // Draw vias: yellow
        for r in 0..goals {
            for &(vy, vx) in &rd.vias[r] {
                img.put_pixel(vx as u32, vy as u32, Rgb([255, 255, 0]));
            }
        }

        let img_path = format!("{output_dir}/map_{map_idx:04}_layer_{l:02}.png");
        img.save(&img_path)?;
    }

    Ok(())
}

// ==================== Main ====================

fn main() {
    let args = Args::parse();

    if args.help_flag {
        print_help();
        return;
    }

    if args.checkpoints < 2 {
        eprintln!("Error: checkpoints (-c) must be >= 2");
        std::process::exit(1);
    }

    if args.goals == 0 {
        eprintln!("Error: goals (-g) must be >= 1");
        std::process::exit(1);
    }

    let num_threads = args.threads.unwrap_or_else(|| {
        (std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) - 1).max(1)
    });

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap_or_else(|_| {
            eprintln!("Warning: could not set thread count, using default");
        });

    let total_maps = args.num;
    let width = args.width;
    let height = args.height;
    let layers = args.layers;
    let goals = args.goals;
    let checkpoints = args.checkpoints;
    let via = args.via;
    let output_path = args.output.clone();
    let render_dir = args.render.clone();

    eprintln!(
        "Generating {total_maps} maze(s): {width}x{height}, layers={layers}, goals={goals}, checkpoints={checkpoints}, via={via}, threads={num_threads}"
    );

    let results: Vec<(MazeMap, Vec<LayerRouteData>)> = (0..total_maps)
        .into_par_iter()
        .map(|i| {
            let seed = (i as u64).wrapping_mul(0xDEADBEEF).wrapping_add(0xCAFE);
            let (map, data) = generate_map(seed, width, height, layers, goals, checkpoints, via);
            if i%1000 == 0 {
                eprintln!("  Generated map {}/{}", i + 1, total_maps);
            }
            (map, data)
        })
        .collect();

    let maps: Vec<MazeMap> = results
        .iter()
        .map(|(m, _)| MazeMap {
            puzzle: m.puzzle.clone(),
            solution: m.solution.clone(),
        })
        .collect();

    eprintln!("Saving to {output_path}...");
    if let Err(e) = save_maps(&maps, width, height, layers, goals, &output_path) {
        eprintln!("Error saving maps: {e}");
        std::process::exit(1);
    }
    eprintln!("Saved {total_maps} map(s) to {output_path}");

    for (i, (map, _data)) in results.iter().enumerate() {
        let errors = validate_map(map, width, height, layers, goals, checkpoints, via);
        if !errors.is_empty() {
            eprintln!("WARNING: Map {i} validation issues:");
            for err in &errors {
                eprintln!("  {err}");
            }
        }
    }

    if let Some(ref dir) = render_dir {
        eprintln!("Rendering images to {dir}...");
        for (i, (map, data)) in results.iter().enumerate() {
            if let Err(e) = render_map(map, data, width, height, layers, goals, dir, i) {
                eprintln!("Error rendering map {i}: {e}");
            }
        }
        eprintln!("Rendering complete.");
    }
}

fn print_help() {
    println!(
        r"Multi-goal multi-layer non-intersecting maze map generator

Usage: generator [OPTIONS]

Options:
  -w <width>       Width of the maze map (default: 64)
  -h <height>      Height of the maze map (default: 64)
  -l <layer>       Number of layers (default: 2)
  -g <goal>        Number of distinct routes (default: 2)
  -c <checkpoint>  Checkpoints per route, >= 2 (default: 2)
  -n <num>         Number of maps to generate (default: 5)
  -o <output>      Output safetensors path (default: maze.safetensors)
  -t <thread>      Number of threads (default: cpu cores - 1)
  -r <dir>         Render solution images to directory
  -v <via>         Target number of forced vias per route (default: 1)
  --help           Print this help"
    );
}