use crate::types::MazeMap;

pub fn validate_map(
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
    }

    for r in 0..goals {
        let mut total_cp = 0usize;
        for l in 0..layers {
            let cp_offset = l * (goals + 1) * channel_size + (r + 1) * channel_size;
            total_cp += (0..channel_size).filter(|&i| map.puzzle[cp_offset + i] == 1).count();
        }
        if total_cp < checkpoints {
            errors.push(format!("Route {r}: only {total_cp} checkpoint cells (expected >= {checkpoints})"));
        }
    }

    for l in 0..layers {
        let sol_offset = l * channel_size;
        let sol_cells = (0..channel_size).filter(|&i| map.solution[sol_offset + i] == 1).count();
        if via_count > 0 && layers > 1 && sol_cells == 0 {
            errors.push(format!("Layer {l}: no solution path cells"));
        }
    }

    errors
}
