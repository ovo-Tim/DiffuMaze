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

    let mut regions = vec![HashSet::new(); parts];
    let mut unclaimed = cells.clone();
    let target = (cells.len() / parts).max(1);
    let mut seed = *cells.iter().choose(rng).unwrap();

    for (part, region) in regions.iter_mut().enumerate() {
        if unclaimed.is_empty() {
            break;
        }
        if !unclaimed.contains(&seed) {
            seed = *unclaimed.iter().choose(rng).unwrap();
        }

        let limit = if part == parts - 1 { unclaimed.len() } else { target };
        let mut queue = VecDeque::new();
        queue.push_back(seed);

        while let Some(cell) = queue.pop_front() {
            if !unclaimed.remove(&cell) {
                continue;
            }
            region.insert(cell);
            if region.len() >= limit {
                break;
            }

            let (mi, mj) = cell;
            let mut neighbors = Vec::new();
            for (dy, dx) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                let ni = mi as i32 + dy;
                let nj = mj as i32 + dx;
                if ni >= 0 && nj >= 0 {
                    let next = (ni as usize, nj as usize);
                    if unclaimed.contains(&next) {
                        neighbors.push(next);
                    }
                }
            }
            neighbors.shuffle(rng);
            for next in neighbors {
                queue.push_back(next);
            }
        }

        seed = region
            .iter()
            .flat_map(|&(mi, mj)| {
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
            })
            .find(|cell| unclaimed.contains(cell))
            .or_else(|| unclaimed.iter().copied().next())
            .unwrap_or(seed);
    }

    if !unclaimed.is_empty() {
        for cell in unclaimed {
            let (best_idx, _) = regions
                .iter()
                .enumerate()
                .filter(|(_, region)| !region.is_empty())
                .map(|(idx, region)| {
                    let min_dist = region
                        .iter()
                        .map(|&owned| {
                            let dy = cell.0 as i64 - owned.0 as i64;
                            let dx = cell.1 as i64 - owned.1 as i64;
                            dy * dy + dx * dx
                        })
                        .min()
                        .unwrap_or(0);
                    (idx, min_dist)
                })
                .min_by_key(|&(_, dist)| dist)
                .unwrap();
            regions[best_idx].insert(cell);
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
                    bridge_path = bridge_between_regions(&base_regions[r], prev_region, next_region);
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
                if layer_grid[l][y][x] {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn via_count_per_route(data: &[LayerRouteData], goals: usize) -> Vec<usize> {
        (0..goals)
            .map(|r| {
                data.iter()
                    .flat_map(|layer| layer.vias[r].iter().copied())
                    .collect::<HashSet<_>>()
                    .len()
            })
            .collect()
    }

    fn via_on_solution(
        map: &MazeMap,
        data: &[LayerRouteData],
        width: usize,
        height: usize,
        layers: usize,
        goals: usize,
    ) -> Vec<bool> {
        let channel_size = height * width;
        let mut result = vec![false; goals];
        for r in 0..goals {
            let any_via = data.iter().any(|layer| !layer.vias[r].is_empty());
            if !any_via {
                continue;
            }
            let mut via_cells_on_path = 0usize;
            let mut _via_cells_total = 0usize;
            for l in 0..layers {
                let sol_offset = l * channel_size;
                let cp_offset = l * (goals + 1) * channel_size + (r + 1) * channel_size;
                for &(vy, vx) in &data[l].vias[r] {
                    _via_cells_total += 1;
                    if map.solution[sol_offset + vy * width + vx] == 1 {
                        via_cells_on_path += 1;
                    }
                    if map.puzzle[cp_offset + vy * width + vx] == 1 {
                        via_cells_on_path += 1;
                    }
                }
            }
            result[r] = via_cells_on_path > 0;
        }
        result
    }

    fn route_markers(
        data: &[LayerRouteData],
        layers: usize,
        goals: usize,
        route: usize,
        include_vias: bool,
    ) -> Vec<(usize, usize, usize)> {
        let mut markers = Vec::new();
        for l in 0..layers {
            for &(y, x) in &data[l].checkpoints[route] {
                markers.push((l, y, x));
            }
            if include_vias {
                for &(y, x) in &data[l].vias[route] {
                    markers.push((l, y, x));
                }
            }
        }
        assert!(route < goals);
        markers
    }

    fn route_via_coords(
        data: &[LayerRouteData],
        layers: usize,
        route: usize,
    ) -> HashSet<(usize, usize)> {
        let mut vias = HashSet::new();
        for l in 0..layers {
            for &(y, x) in &data[l].vias[route] {
                vias.insert((y, x));
            }
        }
        vias
    }

    fn route_reaches_all_markers(
        map: &MazeMap,
        data: &[LayerRouteData],
        width: usize,
        height: usize,
        layers: usize,
        goals: usize,
        route: usize,
        include_vias: bool,
        removed_via: Option<(usize, usize)>,
    ) -> bool {
        let channel_size = height * width;
        let markers = route_markers(data, layers, goals, route, include_vias);
        if markers.len() <= 1 {
            return true;
        }

        let marker_set: HashSet<(usize, usize, usize)> = markers.iter().copied().collect();
        let via_coords = route_via_coords(data, layers, route);
        let start = markers[0];

        let is_open = |l: usize, y: usize, x: usize| {
            let layer_offset = l * (goals + 1) * channel_size;
            map.puzzle[layer_offset + y * width + x] == 0
                && data[l].route_owner[y * width + x] as usize == route
        };

        if !is_open(start.0, start.1, start.2) {
            return false;
        }

        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        seen.insert(start);
        queue.push_back(start);

        while let Some((l, y, x)) = queue.pop_front() {
            for (dy, dx) in &[(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
                let ny = y as i32 + dy;
                let nx = x as i32 + dx;
                if ny >= 0 && nx >= 0 && (ny as usize) < height && (nx as usize) < width {
                    let next = (l, ny as usize, nx as usize);
                    if is_open(next.0, next.1, next.2) && seen.insert(next) {
                        queue.push_back(next);
                    }
                }
            }

            if via_coords.contains(&(y, x)) && removed_via != Some((y, x)) {
                for next_l in 0..layers {
                    if next_l != l
                        && data[next_l].vias[route].contains(&(y, x))
                        && is_open(next_l, y, x)
                    {
                        let next = (next_l, y, x);
                        if seen.insert(next) {
                            queue.push_back(next);
                        }
                    }
                }
            }
        }

        marker_set.iter().all(|marker| seen.contains(marker))
    }

    fn assert_all_route_markers_connected(
        map: &MazeMap,
        data: &[LayerRouteData],
        width: usize,
        height: usize,
        layers: usize,
        goals: usize,
    ) {
        for r in 0..goals {
            assert!(
                route_reaches_all_markers(map, data, width, height, layers, goals, r, true, None),
                "Route {r}: not all checkpoints/vias are connected in the puzzle graph"
            );
        }
    }

    fn assert_all_vias_essential(
        map: &MazeMap,
        data: &[LayerRouteData],
        width: usize,
        height: usize,
        layers: usize,
        goals: usize,
    ) {
        for r in 0..goals {
            assert!(
                route_reaches_all_markers(map, data, width, height, layers, goals, r, false, None),
                "Route {r}: checkpoints are not connected before via-essential check"
            );
            for via in route_via_coords(data, layers, r) {
                assert!(
                    !route_reaches_all_markers(
                        map,
                        data,
                        width,
                        height,
                        layers,
                        goals,
                        r,
                        false,
                        Some(via)
                    ),
                    "Route {r}: via at {:?} is bypassable",
                    via
                );
            }
        }
    }

    #[test]
    fn test_multiple_vias_with_enough_checkpoints() {
        let (map, data) = generate_map(42, 64, 64, 2, 2, 4, 3);
        assert_all_route_markers_connected(&map, &data, 64, 64, 2, 2);
        assert_all_vias_essential(&map, &data, 64, 64, 2, 2);
        let counts = via_count_per_route(&data, 2);
        for (r, &count) in counts.iter().enumerate() {
            assert!(
                count >= 2,
                "Route {}: expected >= 2 vias with -c 4 -v 3, got {}",
                r,
                count
            );
        }
        let on_path = via_on_solution(&map, &data, 64, 64, 2, 2);
        for (r, &p) in on_path.iter().enumerate() {
            let count = counts[r];
            if count > 0 {
                assert!(p, "Route {}: via not on solution path", r);
            }
        }
    }

    #[test]
    fn test_few_checkpoints_many_vias() {
        let (map, data) = generate_map(42, 64, 64, 2, 2, 2, 3);
        assert_all_route_markers_connected(&map, &data, 64, 64, 2, 2);
        assert_all_vias_essential(&map, &data, 64, 64, 2, 2);
        let counts = via_count_per_route(&data, 2);
        for (r, &count) in counts.iter().enumerate() {
            assert_eq!(
                count, 3,
                "Route {}: expected 3 vias with -c 2 -v 3, got {}",
                r, count
            );
        }
    }

    #[test]
    fn test_via_count_matches_gaps() {
        for checkpoints in 3..=6 {
            let via_target = checkpoints - 1;
            let (_map, data) = generate_map(99, 64, 64, 2, 2, checkpoints, via_target);
            let counts = via_count_per_route(&data, 2);
            for (r, &count) in counts.iter().enumerate() {
                assert!(
                    count >= 1,
                    "Route {}: expected >= 1 via with c={} v={}, got {}",
                    r,
                    checkpoints,
                    via_target,
                    count
                );
                let expected = if checkpoints >= via_target + 1 {
                    via_target
                } else {
                    checkpoints - 1
                };
                assert!(
                    count <= expected,
                    "Route {}: expected <= {} vias, got {}",
                    r,
                    expected,
                    count
                );
            }
        }
    }

    #[test]
    fn test_no_vias_with_single_layer() {
        let (_, data) = generate_map(42, 64, 64, 1, 2, 4, 3);
        let counts = via_count_per_route(&data, 2);
        for (r, &count) in counts.iter().enumerate() {
            assert_eq!(
                count, 0,
                "Route {}: expected 0 vias with 1 layer, got {}",
                r, count
            );
        }
    }

    #[test]
    fn test_no_vias_with_via_zero() {
        let (_, data) = generate_map(42, 64, 64, 2, 2, 4, 0);
        let counts = via_count_per_route(&data, 2);
        for (r, &count) in counts.iter().enumerate() {
            assert_eq!(
                count, 0,
                "Route {}: expected 0 vias with -v 0, got {}",
                r, count
            );
        }
    }

    #[test]
    fn test_vias_on_solution_path() {
        let (map, data) = generate_map(42, 64, 64, 2, 2, 4, 3);
        let on_path = via_on_solution(&map, &data, 64, 64, 2, 2);
        for (r, &p) in on_path.iter().enumerate() {
            let count = via_count_per_route(&data, 2)[r];
            if count > 0 {
                assert!(
                    p,
                    "Route {}: via exists but not on solution path or puzzle checkpoint",
                    r
                );
            }
        }
    }

    #[test]
    fn test_three_layers_multiple_vias() {
        let (map, data) = generate_map(42, 64, 64, 3, 2, 4, 2);
        assert_all_route_markers_connected(&map, &data, 64, 64, 3, 2);
        assert_all_vias_essential(&map, &data, 64, 64, 3, 2);
        let counts = via_count_per_route(&data, 2);
        for (r, &count) in counts.iter().enumerate() {
            assert!(
                count >= 1,
                "Route {}: expected >= 1 via with 3 layers -c 4 -v 2, got {}",
                r,
                count
            );
        }
        let on_path = via_on_solution(&map, &data, 64, 64, 3, 2);
        for (r, &p) in on_path.iter().enumerate() {
            let count = counts[r];
            if count > 0 {
                assert!(p, "Route {}: via not on solution path", r);
            }
        }
    }

    #[test]
    fn test_via_count_increases_with_checkpoints() {
        let cases = vec![
            (2usize, 1usize, 1usize),
            (2, 2, 2),
            (2, 3, 3),
            (3, 2, 2),
            (4, 3, 3),
            (5, 3, 3),
            (6, 4, 4),
        ];
        for (c, v, expected_vias) in &cases {
            let (_, data) = generate_map(42, 64, 64, 2, 2, *c, *v);
            let counts = via_count_per_route(&data, 2);
            for (r, &count) in counts.iter().enumerate() {
                assert_eq!(
                    count, *expected_vias,
                    "Route {}: -c {} -v {} -> expected {} vias, got {}",
                    r, c, v, expected_vias, count
                );
            }
        }
    }
}
