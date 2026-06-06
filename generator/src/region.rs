use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::{HashSet, VecDeque};

pub fn partition_regions(
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
