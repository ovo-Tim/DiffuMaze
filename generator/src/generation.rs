use crate::checkpoint::pick_checkpoints_in_region;
use crate::maze::{find_path_in_maze, generate_maze_dfs, grid_to_maze, maze_to_grid};
use crate::region::partition_regions;
use crate::types::{LayerRouteData, MazeMap};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashSet;

pub fn generate_map(
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
    let mut route_vias: Vec<Vec<(usize, usize)>> = vec![Vec::new(); goals];

    if via_count > 0 && layers > 1 {
        for r in 0..goals {
            let region_cells: Vec<(usize, usize)> = regions[r].iter().copied().collect();
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
    let mut route_layer_checkpoints: Vec<Vec<Vec<(usize, usize)>>> =
        vec![vec![Vec::new(); layers]; goals];

    for r in 0..goals {
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

        let mut cp_idx = 0;
        let mut all_cps = Vec::new();

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
    let mut layer_solution_cells: Vec<Vec<HashSet<(usize, usize)>>> =
        vec![vec![HashSet::new(); goals]; layers];
    let mut layer_grid: Vec<Vec<Vec<bool>>> =
        vec![vec![vec![false; width]; height]; layers];
    let mut layer_cell_owner: Vec<Vec<Vec<usize>>> =
        vec![vec![vec![0; width]; height]; layers];

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

    for r in 0..goals {
        for &(mi, mj) in &route_vias[r] {
            let (y, x) = maze_to_grid(mi, mj);
            for l in 0..layers {
                layer_grid[l][y][x] = true;
                layer_cell_owner[l][y][x] = r;
            }
        }
    }

    for r in 0..goals {
        for &(mi, mj) in &route_vias[r] {
            for l in 0..layers {
                let edges = &layer_edges[l][r];
                let connected = edges.iter().any(|&(a, b)| {
                    a == (mi, mj) || b == (mi, mj)
                });

                if !connected && regions[r].contains(&(mi, mj)) {
                    let (vy, vx) = maze_to_grid(mi, mj);
                    for (dy, dx) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                        let ni = mi as i32 + dy;
                        let nj = mj as i32 + dx;
                        if ni >= 0 && nj >= 0 && (ni as usize) < maze_h && (nj as usize) < maze_w {
                            let n = (ni as usize, nj as usize);
                            if regions[r].contains(&n) {
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
    for r in 0..goals {
        let mut waypoints: Vec<(usize, (usize, usize))> = Vec::new();

        let cps_per_layer = &route_layer_checkpoints[r];

        let mut all_cp_ordered: Vec<(usize, usize, usize)> = Vec::new();
        for l in 0..layers {
            for &(gy, gx) in &cps_per_layer[l] {
                let (mi, mj) = grid_to_maze(gy, gx);
                all_cp_ordered.push((l, mi, mj));
            }
        }

        if all_cp_ordered.is_empty() {
            continue;
        }

        waypoints.push((all_cp_ordered[0].0, (all_cp_ordered[0].1, all_cp_ordered[0].2)));

        for i in 1..all_cp_ordered.len() {
            let prev_layer = all_cp_ordered[i - 1].0;
            let curr_layer = all_cp_ordered[i].0;

            if prev_layer != curr_layer && !route_vias[r].is_empty() {
                let via_idx = (i - 1) % route_vias[r].len();
                waypoints.push((prev_layer, route_vias[r][via_idx]));
                waypoints.push((curr_layer, route_vias[r][via_idx]));
            }

            waypoints.push((curr_layer, (all_cp_ordered[i].1, all_cp_ordered[i].2)));
        }

        for w in 0..waypoints.len().saturating_sub(1) {
            let (layer_a, cell_a) = waypoints[w];
            let (layer_b, cell_b) = waypoints[w + 1];

            if layer_a != layer_b {
                continue;
            }

            let layer = layer_a;
            let edges = &layer_edges[layer][r];

            let path = find_path_in_maze(cell_a, cell_b, edges, &regions[r]);

            for &(mi, mj) in &path {
                let (y, x) = maze_to_grid(mi, mj);
                layer_solution_cells[layer][r].insert((y, x));
            }
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

        for y in 0..height {
            for x in 0..width {
                if !layer_grid[l][y][x] {
                    puzzle[layer_offset + y * width + x] = 1;
                }
            }
        }

        for r in 0..goals {
            let cp_offset = layer_offset + (r + 1) * channel_size;
            for &(cy, cx) in &route_layer_checkpoints[r][l] {
                puzzle[cp_offset + cy * width + cx] = 1;
            }
            for &(mi, mj) in &route_vias[r] {
                let (vy, vx) = maze_to_grid(mi, mj);
                if layer_grid[l][vy][vx] {
                    puzzle[cp_offset + vy * width + vx] = 1;
                }
            }
        }

        for r in 0..goals {
            for &(y, x) in &layer_solution_cells[l][r] {
                solution[sol_offset + y * width + x] = 1;
            }
        }

        let mut route_owner = vec![goals as u8; channel_size];
        for y in 0..height {
            for x in 0..width {
                if layer_grid[l][y][x] {
                    route_owner[y * width + x] = layer_cell_owner[l][y][x] as u8;
                }
            }
        }

        let mut layer_checkpoints: Vec<Vec<(usize, usize)>> = Vec::new();
        for r in 0..goals {
            layer_checkpoints.push(route_layer_checkpoints[r][l].clone());
        }

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
