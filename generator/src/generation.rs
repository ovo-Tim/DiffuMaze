use crate::checkpoint::pick_checkpoints_in_region;
use crate::maze::{find_path_in_maze, generate_maze_dfs, maze_to_grid};
use crate::region::partition_regions;
use crate::types::{LayerRouteData, MazeMap};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet, VecDeque};

fn partition_region_chain(
    rng: &mut StdRng,
    cells: &HashSet<(usize, usize)>,
    parts: usize,
) -> Vec<HashSet<(usize, usize)>> {
    if parts == 0 {
        return Vec::new();
    }
    if cells.is_empty() {
        return vec![HashSet::new(); parts];
    }

    let neighbors = |(mi, mj): (usize, usize)| {
        [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)]
            .into_iter()
            .filter_map(move |(dy, dx)| {
                let ni = mi as i32 + dy;
                let nj = mj as i32 + dx;
                if ni >= 0 && nj >= 0 {
                    Some((ni as usize, nj as usize))
                } else {
                    None
                }
            })
    };

    let bfs = |start: (usize, usize)| {
        let mut queue = VecDeque::new();
        let mut parent: HashMap<(usize, usize), Option<(usize, usize)>> = HashMap::new();
        let mut last = start;
        parent.insert(start, None);
        queue.push_back(start);

        while let Some(cell) = queue.pop_front() {
            last = cell;
            for next in neighbors(cell) {
                if cells.contains(&next) && !parent.contains_key(&next) {
                    parent.insert(next, Some(cell));
                    queue.push_back(next);
                }
            }
        }

        (last, parent)
    };

    let start = *cells.iter().choose(rng).unwrap();
    let (end_a, _) = bfs(start);
    let (end_b, parent) = bfs(end_a);

    let mut spine = Vec::new();
    let mut current = end_b;
    loop {
        spine.push(current);
        match parent[&current] {
            Some(prev) => current = prev,
            None => break,
        }
    }
    spine.reverse();

    let mut regions = vec![HashSet::new(); parts];
    let mut claimed = HashSet::new();
    let mut queue = VecDeque::new();

    for (idx, &cell) in spine.iter().enumerate() {
        let region_idx = (idx * parts / spine.len()).min(parts - 1);
        regions[region_idx].insert(cell);
        claimed.insert(cell);
        queue.push_back((region_idx, cell));
    }

    while let Some((region_idx, cell)) = queue.pop_front() {
        let mut next_cells: Vec<(usize, usize)> = neighbors(cell)
            .filter(|next| cells.contains(next) && !claimed.contains(next))
            .collect();
        next_cells.shuffle(rng);

        for next in next_cells {
            if claimed.insert(next) {
                regions[region_idx].insert(next);
                queue.push_back((region_idx, next));
            }
        }
    }

    regions
}

fn bridge_between_regions(
    cells: &HashSet<(usize, usize)>,
    start: &HashSet<(usize, usize)>,
    end: &HashSet<(usize, usize)>,
) -> Option<Vec<(usize, usize)>> {
    if start.is_empty() || end.is_empty() {
        return None;
    }

    let mut queue = VecDeque::new();
    let mut parent: HashMap<(usize, usize), Option<(usize, usize)>> = HashMap::new();

    for &cell in start {
        if cells.contains(&cell) {
            parent.insert(cell, None);
            queue.push_back(cell);
        }
    }

    while let Some(cell) = queue.pop_front() {
        if end.contains(&cell) {
            let mut path = Vec::new();
            let mut current = cell;
            loop {
                path.push(current);
                match parent[&current] {
                    Some(prev) => current = prev,
                    None => break,
                }
            }
            path.reverse();
            return Some(path);
        }

        let (mi, mj) = cell;
        for (dy, dx) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let ni = mi as i32 + dy;
            let nj = mj as i32 + dx;
            if ni >= 0 && nj >= 0 {
                let next = (ni as usize, nj as usize);
                if cells.contains(&next) && !parent.contains_key(&next) {
                    parent.insert(next, Some(cell));
                    queue.push_back(next);
                }
            }
        }
    }

    None
}

fn add_background_filler(
    rng: &mut StdRng,
    layer_grid: &mut [Vec<Vec<bool>>],
    layer_cell_owner: &mut [Vec<Vec<usize>>],
    width: usize,
    height: usize,
    maze_w: usize,
    maze_h: usize,
    goals: usize,
) {
    for l in 0..layer_grid.len() {
        let route_grid = layer_grid[l].clone();
        let mut safe_grid = vec![vec![false; width]; height];

        for y in 0..height {
            for x in 0..width {
                if route_grid[y][x] {
                    continue;
                }

                let touches_route = [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)]
                    .into_iter()
                    .any(|(dy, dx)| {
                        let ny = y as i32 + dy;
                        let nx = x as i32 + dx;
                        ny >= 0
                            && nx >= 0
                            && (ny as usize) < height
                            && (nx as usize) < width
                            && route_grid[ny as usize][nx as usize]
                    });

                safe_grid[y][x] = !touches_route;
            }
        }

        let mut safe_cells = HashSet::new();
        for mi in 0..maze_h {
            for mj in 0..maze_w {
                let (y, x) = maze_to_grid(mi, mj);
                if y < height && x < width && safe_grid[y][x] {
                    safe_cells.insert((mi, mj));
                }
            }
        }

        let mut unvisited = safe_cells.clone();
        while let Some(start) = unvisited.iter().copied().choose(rng) {
            let mut component = HashSet::new();
            let mut queue = VecDeque::new();
            unvisited.remove(&start);
            component.insert(start);
            queue.push_back(start);

            while let Some((mi, mj)) = queue.pop_front() {
                for (dy, dx) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                    let ni = mi as i32 + dy;
                    let nj = mj as i32 + dx;
                    if ni >= 0 && nj >= 0 && (ni as usize) < maze_h && (nj as usize) < maze_w {
                        let next = (ni as usize, nj as usize);
                        if safe_cells.contains(&next) && unvisited.remove(&next) {
                            component.insert(next);
                            queue.push_back(next);
                        }
                    }
                }
            }

            for &(mi, mj) in &component {
                let (y, x) = maze_to_grid(mi, mj);
                if !layer_grid[l][y][x] {
                    layer_grid[l][y][x] = true;
                    layer_cell_owner[l][y][x] = goals;
                }
            }

            let filler_edges = generate_maze_dfs(rng, &component, maze_w, maze_h);
            for &(a, b) in &filler_edges {
                let (y1, x1) = maze_to_grid(a.0, a.1);
                let (y2, x2) = maze_to_grid(b.0, b.1);
                let wy = (y1 + y2) / 2;
                let wx = (x1 + x2) / 2;
                if wy < height && wx < width && safe_grid[wy][wx] && !layer_grid[l][wy][wx] {
                    layer_grid[l][wy][wx] = true;
                    layer_cell_owner[l][wy][wx] = goals;
                }
            }
        }
    }
}

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

    let total_segments = if via_count > 0 && layers > 1 {
        via_count + 1
    } else {
        1
    };
    let segment_layers: Vec<usize> = (0..total_segments)
        .map(|seg| if layers > 1 { seg % layers } else { 0 })
        .collect();

    // Step 1: Partition goals once, then subdivide each goal footprint by route segment.
    let base_regions = partition_regions(&mut rng, maze_w, maze_h, goals);
    let mut route_segment_regions: Vec<Vec<HashSet<(usize, usize)>>> =
        vec![vec![HashSet::new(); total_segments]; goals];

    for r in 0..goals {
        let regions = partition_region_chain(&mut rng, &base_regions[r], total_segments);
        for (seg, region) in regions.into_iter().enumerate() {
            route_segment_regions[r][seg] = region;
        }
    }

    // Step 2: Place vias for each route at segment boundaries.
    let mut route_vias: Vec<Vec<(usize, usize)>> = vec![Vec::new(); goals];

    if via_count > 0 && layers > 1 {
        for r in 0..goals {
            let mut chosen_vias = Vec::new();
            let mut used = HashSet::new();

            for boundary in 0..via_count {
                let prev_region = &route_segment_regions[r][boundary];
                let next_region = &route_segment_regions[r][boundary + 1];
                let mut candidates = Vec::new();
                let mut bridge_path = None;

                for &cell in prev_region {
                    let (mi, mj) = cell;
                    if [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)]
                        .into_iter()
                        .any(|(dy, dx)| {
                            let ni = mi as i32 + dy;
                            let nj = mj as i32 + dx;
                            ni >= 0 && nj >= 0 && next_region.contains(&(ni as usize, nj as usize))
                        })
                    {
                        candidates.push(cell);
                    }
                }

                for &cell in next_region {
                    let (mi, mj) = cell;
                    if [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)]
                        .into_iter()
                        .any(|(dy, dx)| {
                            let ni = mi as i32 + dy;
                            let nj = mj as i32 + dx;
                            ni >= 0 && nj >= 0 && prev_region.contains(&(ni as usize, nj as usize))
                        })
                    {
                        candidates.push(cell);
                    }
                }

                candidates.retain(|cell| !used.contains(cell));
                candidates.sort_unstable();
                candidates.dedup();

                if candidates.is_empty() {
                    let mut bridge_cells = prev_region.clone();
                    bridge_cells.extend(next_region.iter().copied());
                    bridge_path = bridge_between_regions(&bridge_cells, prev_region, next_region);
                    if let Some(path) = &bridge_path {
                        candidates = path
                            .iter()
                            .filter(|&&cell| !used.contains(&cell))
                            .copied()
                            .collect();
                    }
                }

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

                if let Some(&via) = chosen_vias.last() {
                    if let Some(path) = &bridge_path {
                        if let Some(pos) = path.iter().position(|&cell| cell == via) {
                            for &cell in &path[..=pos] {
                                route_segment_regions[r][boundary].insert(cell);
                            }
                            for &cell in &path[pos..] {
                                route_segment_regions[r][boundary + 1].insert(cell);
                            }
                        }
                    }
                    route_segment_regions[r][boundary].insert(via);
                    route_segment_regions[r][boundary + 1].insert(via);
                }
            }
            route_vias[r] = chosen_vias;
        }
    }

    // Step 3: Generate a separate maze for each route segment region.
    let mut route_segment_edges: Vec<Vec<HashSet<((usize, usize), (usize, usize))>>> =
        vec![vec![HashSet::new(); total_segments]; goals];
    for r in 0..goals {
        for seg in 0..total_segments {
            route_segment_edges[r][seg] =
                generate_maze_dfs(&mut rng, &route_segment_regions[r][seg], maze_w, maze_h);
        }
    }

    // Step 4: Assign checkpoints to layers for each route
    let mut route_layer_checkpoints: Vec<Vec<Vec<(usize, usize)>>> =
        vec![vec![Vec::new(); layers]; goals];
    let mut route_segments: Vec<Vec<(usize, Vec<(usize, usize)>)>> = vec![Vec::new(); goals];

    for r in 0..goals {
        let total_segments = if via_count > 0 && layers > 1 {
            via_count + 1
        } else {
            1
        };

        let via_set: HashSet<(usize, usize)> = route_vias[r].iter().copied().collect();

        let mut seg_cps: Vec<Vec<(usize, usize)>> = vec![Vec::new(); total_segments];

        if checkpoints == 0 || total_segments == 1 {
            if checkpoints > 0 {
                let cps = pick_checkpoints_in_region(
                    &mut rng,
                    &route_segment_regions[r][0],
                    checkpoints,
                    &via_set,
                );
                for &(mi, mj) in &cps {
                    seg_cps[0].push((mi, mj));
                }
            }
        } else if checkpoints == 1 {
            let cps =
                pick_checkpoints_in_region(&mut rng, &route_segment_regions[r][0], 1, &via_set);
            for &(mi, mj) in &cps {
                seg_cps[0].push((mi, mj));
            }
        } else {
            let mut seg_counts = vec![0usize; total_segments];
            seg_counts[0] = 1;
            seg_counts[total_segments - 1] = 1;
            let remaining = checkpoints - 2;
            for i in 0..remaining {
                seg_counts[(i + 1) % total_segments] += 1;
            }
            for seg in 0..total_segments {
                if seg_counts[seg] == 0 {
                    continue;
                }
                let cps = pick_checkpoints_in_region(
                    &mut rng,
                    &route_segment_regions[r][seg],
                    seg_counts[seg],
                    &via_set,
                );
                for &(mi, mj) in &cps {
                    seg_cps[seg].push((mi, mj));
                }
            }
        }

        for seg in 0..total_segments {
            let layer = if layers > 1 { seg % layers } else { 0 };
            for &(mi, mj) in &seg_cps[seg] {
                let (y, x) = maze_to_grid(mi, mj);
                route_layer_checkpoints[r][layer].push((y, x));
            }
        }
        route_segments[r] = seg_cps
            .iter()
            .enumerate()
            .map(|(seg, cps)| {
                let layer = if layers > 1 { seg % layers } else { 0 };
                (layer, cps.clone())
            })
            .collect();
    }

    // Step 5: Build the multi-layer route (solution path)
    let mut layer_solution_cells: Vec<Vec<HashSet<(usize, usize)>>> =
        vec![vec![HashSet::new(); goals]; layers];
    let mut layer_grid: Vec<Vec<Vec<bool>>> = vec![vec![vec![false; width]; height]; layers];
    let mut layer_cell_owner: Vec<Vec<Vec<usize>>> = vec![vec![vec![0; width]; height]; layers];

    for r in 0..goals {
        for seg in 0..total_segments {
            let l = segment_layers[seg];
            let edges = &route_segment_edges[r][seg];

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

            for &(mi, mj) in &route_segment_regions[r][seg] {
                let (y, x) = maze_to_grid(mi, mj);
                layer_grid[l][y][x] = true;
                layer_cell_owner[l][y][x] = r;
            }
        }
    }

    // Step 6: Construct the full route path for each route
    let mut used_vias: Vec<HashSet<(usize, usize)>> = vec![HashSet::new(); goals];

    for r in 0..goals {
        let segs = &route_segments[r];
        if segs.is_empty() {
            continue;
        }

        let has_any_cp = segs.iter().any(|(_, cps)| !cps.is_empty());
        if !has_any_cp {
            continue;
        }

        let total_segs = segs.len();
        let mut waypoints: Vec<(usize, usize, (usize, usize))> = Vec::new();
        let mut via_used = 0usize;

        for seg_idx in 0..total_segs {
            let layer = segs[seg_idx].0;

            if seg_idx > 0 {
                let prev_layer = segs[seg_idx - 1].0;
                if prev_layer != layer && !route_vias[r].is_empty() {
                    let via = route_vias[r][via_used % route_vias[r].len()];
                    used_vias[r].insert(via);
                    waypoints.push((seg_idx - 1, prev_layer, via));
                    waypoints.push((seg_idx, layer, via));
                    via_used += 1;
                }
            }

            for &cp in &segs[seg_idx].1 {
                waypoints.push((seg_idx, layer, cp));
            }
        }

        for w in 0..waypoints.len().saturating_sub(1) {
            let (seg_a, layer_a, cell_a) = waypoints[w];
            let (seg_b, layer_b, cell_b) = waypoints[w + 1];

            if layer_a != layer_b || seg_a != seg_b {
                continue;
            }

            let layer = layer_a;
            let edges = &route_segment_edges[r][seg_a];

            let path = find_path_in_maze(cell_a, cell_b, edges, &route_segment_regions[r][seg_a]);

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

    for r in 0..goals {
        route_vias[r].retain(|v| used_vias[r].contains(v));
    }

    for r in 0..goals {
        for (boundary, &(mi, mj)) in route_vias[r].iter().enumerate() {
            for &seg in &[boundary, boundary + 1] {
                if seg >= total_segments {
                    continue;
                }
                let l = segment_layers[seg];
                let (y, x) = maze_to_grid(mi, mj);
                layer_grid[l][y][x] = true;
                layer_cell_owner[l][y][x] = r;

                let edges = &route_segment_edges[r][seg];
                let connected = edges.iter().any(|&(a, b)| a == (mi, mj) || b == (mi, mj));

                if !connected && route_segment_regions[r][seg].contains(&(mi, mj)) {
                    let (vy, vx) = maze_to_grid(mi, mj);
                    for (dy, dx) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                        let ni = mi as i32 + dy;
                        let nj = mj as i32 + dx;
                        if ni >= 0 && nj >= 0 && (ni as usize) < maze_h && (nj as usize) < maze_w {
                            let n = (ni as usize, nj as usize);
                            if route_segment_regions[r][seg].contains(&n) {
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

    add_background_filler(
        &mut rng,
        &mut layer_grid,
        &mut layer_cell_owner,
        width,
        height,
        maze_w,
        maze_h,
        goals,
    );

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
                if layer_grid[l][vy][vx] && layer_cell_owner[l][vy][vx] == r {
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
                if layer_grid[l][y][x] && layer_cell_owner[l][y][x] == r {
                    layer_vias[r].push((y, x));
                }
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
