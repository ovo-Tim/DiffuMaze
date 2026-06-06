use crate::types::{LayerRouteData, MazeMap};
use std::fs;

const ROUTE_COLORS: [[u8; 3]; 10] = [
    [220, 50, 50],
    [50, 180, 50],
    [50, 80, 220],
    [220, 180, 30],
    [180, 50, 180],
    [50, 180, 180],
    [200, 120, 50],
    [120, 50, 200],
    [50, 200, 100],
    [200, 50, 120],
];

pub fn render_map(
    map: &MazeMap,
    route_data: &[LayerRouteData],
    width: usize,
    height: usize,
    layers: usize,
    goals: usize,
    output_dir: &str,
    map_idx: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use image::{Rgb, RgbImage};

    let channel_size = height * width;
    fs::create_dir_all(output_dir)?;

    for l in 0..layers {
        let mut img = RgbImage::new(width as u32, height as u32);

        let sol_offset = l * channel_size;
        let puzzle_layer_offset = l * (goals + 1) * channel_size;
        let rd = &route_data[l];

        for y in 0..height {
            for x in 0..width {
                if map.puzzle[puzzle_layer_offset + y * width + x] == 1 {
                    img.put_pixel(x as u32, y as u32, Rgb([0, 0, 0]));
                } else {
                    let owner = rd.route_owner[y * width + x] as usize;
                    if owner < goals && map.solution[sol_offset + y * width + x] == 1 {
                        let color = ROUTE_COLORS[owner % ROUTE_COLORS.len()];
                        img.put_pixel(x as u32, y as u32, Rgb(color));
                    } else {
                        img.put_pixel(x as u32, y as u32, Rgb([128, 128, 128]));
                    }
                }
            }
        }

        for r in 0..goals {
            for &(cy, cx) in &rd.checkpoints[r] {
                img.put_pixel(cx as u32, cy as u32, Rgb([255, 255, 255]));
            }
        }

        for r in 0..goals {
            for &(vy, vx) in &rd.vias[r] {
                img.put_pixel(vx as u32, vy as u32, Rgb([255, 255, 0]));
            }
        }

        let img_path = format!("{output_dir}/map_{map_idx:04}_layer_{l:02}.png");
        img.save(&img_path)?;
    }

    Ok(())
}
