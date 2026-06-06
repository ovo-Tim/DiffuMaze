use crate::generation::generate_map;
use crate::types::{LayerRouteData, MazeMap};
use std::collections::{HashSet, VecDeque};

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
    assert!(route < goals);
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
    restrict_to_route_owner: bool,
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
            && (!restrict_to_route_owner || data[l].route_owner[y * width + x] as usize == route)
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
            route_reaches_all_markers(
                map, data, width, height, layers, goals, r, true, true, None
            ),
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
            route_reaches_all_markers(
                map, data, width, height, layers, goals, r, false, true, None
            ),
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
                    true,
                    Some(via)
                ),
                "Route {r}: via at {:?} is bypassable",
                via
            );
        }
    }
}

fn assert_all_vias_essential_on_full_puzzle(
    map: &MazeMap,
    data: &[LayerRouteData],
    width: usize,
    height: usize,
    layers: usize,
    goals: usize,
) {
    for r in 0..goals {
        assert!(
            route_reaches_all_markers(
                map, data, width, height, layers, goals, r, false, false, None
            ),
            "Route {r}: checkpoints are not connected in the full puzzle graph"
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
                    false,
                    Some(via)
                ),
                "Route {r}: via at {:?} is bypassable in the full puzzle graph",
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
    assert_all_vias_essential_on_full_puzzle(&map, &data, 64, 64, 2, 2);
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
    assert_all_vias_essential_on_full_puzzle(&map, &data, 64, 64, 2, 2);
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
fn test_background_filler_opens_unowned_space() {
    let (map, data) = generate_map(42, 64, 64, 2, 2, 2, 3);
    let width = 64;
    let height = 64;
    let layers = 2;
    let goals = 2;
    let channel_size = width * height;
    let mut unowned_open = 0usize;

    for l in 0..layers {
        let layer_offset = l * (goals + 1) * channel_size;
        for y in 0..height {
            for x in 0..width {
                if map.puzzle[layer_offset + y * width + x] == 0
                    && data[l].route_owner[y * width + x] as usize == goals
                {
                    unowned_open += 1;
                }
            }
        }
    }

    assert!(
        unowned_open > channel_size / 10,
        "expected visible background filler, got only {} unowned open cells",
        unowned_open
    );
}

#[test]
fn test_background_filler_preserves_full_puzzle_essential_vias() {
    let (map, data) = generate_map(42, 64, 64, 2, 2, 2, 3);
    assert_all_vias_essential_on_full_puzzle(&map, &data, 64, 64, 2, 2);
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
    assert_all_vias_essential_on_full_puzzle(&map, &data, 64, 64, 3, 2);
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
