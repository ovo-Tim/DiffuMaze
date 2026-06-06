use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "maze-generator", disable_help_flag = true)]
pub struct Args {
    #[arg(short = 'w', default_value = "64")]
    pub width: usize,

    #[arg(short = 'h', default_value = "64")]
    pub height: usize,

    #[arg(short = 'l', default_value = "2")]
    pub layers: usize,

    #[arg(short = 'g', default_value = "2")]
    pub goals: usize,

    #[arg(short = 'c', default_value = "2")]
    pub checkpoints: usize,

    #[arg(short = 'n', default_value = "5")]
    pub num: usize,

    #[arg(short = 'o', default_value = "maze.safetensors")]
    pub output: String,

    #[arg(short = 't')]
    pub threads: Option<usize>,

    #[arg(short = 'r')]
    pub render: Option<String>,

    #[arg(short = 'v', default_value = "1")]
    pub via: usize,

    #[arg(long = "help", action = clap::ArgAction::SetTrue)]
    pub help_flag: bool,
}

pub fn print_help() {
    println!(
        r"Multi-goal multi-layer non-intersecting maze map generator

Usage: generator [OPTIONS]

Options:
  -w <width>       Width of the maze map (default: 64)
  -h <height>      Height of the maze map (default: 64)
  -l <layer>       Number of layers (default: 2)
  -g <goal>        Number of distinct routes (default: 2)
  -c <checkpoint>  Checkpoints per route, >= 2 (default: 2)
  -n <num>         Number of maps to generate (default: 5)
  -o <output>      Output safetensors path (default: maze.safetensors)
  -t <thread>      Number of threads (default: cpu cores - 1)
  -r <dir>         Render solution images to directory
  -v <via>         Target number of forced vias per route (default: 1)
  --help           Print this help"
    );
}
