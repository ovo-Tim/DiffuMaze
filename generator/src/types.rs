pub struct MazeMap {
    pub puzzle: Vec<i8>,
    pub solution: Vec<i8>,
}

#[derive(Clone)]
pub struct LayerRouteData {
    pub route_owner: Vec<u8>,
    pub checkpoints: Vec<Vec<(usize, usize)>>,
    pub vias: Vec<Vec<(usize, usize)>>,
}
