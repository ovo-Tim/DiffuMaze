use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashSet;

pub fn pick_checkpoints_in_region(
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
