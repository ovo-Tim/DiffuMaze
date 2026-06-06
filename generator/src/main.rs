mod cli;
mod checkpoint;
mod generation;
mod io;
mod maze;
mod region;
mod render;
mod types;
mod validation;

use clap::Parser;
use cli::{print_help, Args};
use generation::generate_map;
use io::save_maps;
use rayon::prelude::*;
use render::render_map;
use types::MazeMap;
use validation::validate_map;

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

    let results: Vec<(MazeMap, Vec<_>)> = (0..total_maps)
        .into_par_iter()
        .map(|i| {
            let seed = (i as u64).wrapping_mul(0xDEADBEEF).wrapping_add(0xCAFE);
            let (map, data) = generate_map(seed, width, height, layers, goals, checkpoints, via);
            if i % 1000 == 0 {
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

    for (i, (map, _)) in results.iter().enumerate() {
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
