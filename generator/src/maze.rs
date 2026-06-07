use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet, VecDeque};

pub fn maze_to_grid(mi: usize, mj: usize) -> (usize, usize) {
    (2 * mi + 1, 2 * mj + 1)
}

pub fn generate_maze_dfs(
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

pub fn find_path_in_maze(
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
