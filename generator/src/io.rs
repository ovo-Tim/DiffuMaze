use crate::types::MazeMap;
use std::collections::HashMap;
use std::path::PathBuf;

pub fn save_maps(
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
