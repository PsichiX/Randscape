pub mod diagnostics;

use crate::grid::{FixedGrid, Grid, GridDirection};
use bitvec::vec::BitVec;
use rand::Rng;
use std::{collections::BTreeMap, error::Error};
use vek::Vec2;

#[derive(Debug)]
pub enum WfcError {
    WrongGridSize {
        expected: usize,
        provided: Vec2<usize>,
    },
    PatternSizeIsNotOdd {
        provided: usize,
    },
    PatternSizeIsNotOne {
        provided: usize,
    },
}

impl std::fmt::Display for WfcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongGridSize { expected, provided } => {
                write!(
                    f,
                    "Wrong grid size: expected {}, provided {}",
                    expected, provided
                )
            }
            Self::PatternSizeIsNotOdd { provided } => {
                write!(f, "Pattern size is not odd: {}", provided)
            }
            Self::PatternSizeIsNotOne { provided } => {
                write!(f, "Pattern size is not one: {}", provided)
            }
        }
    }
}

impl Error for WfcError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PatternId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub struct Pattern<const N: usize, T: Copy> {
    id: PatternId,
    grid: FixedGrid<N, N, T>,
    frequency: usize,
    weight: f64,
}

impl<const N: usize, T: Copy> Pattern<N, T> {
    pub fn id(&self) -> PatternId {
        self.id
    }

    pub fn grid(&self) -> &FixedGrid<N, N, T> {
        &self.grid
    }

    pub fn frequency(&self) -> usize {
        self.frequency
    }

    pub fn weight(&self) -> f64 {
        self.weight
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub enum WfcWeightingStrategy {
    #[default]
    Frequency,
    Uniform,
    LogScaling,
    PowerScaling(f32),
    CapMax(usize),
    InverseFrequency,
}

impl WfcWeightingStrategy {
    pub fn weight(&self, frequency: usize) -> f64 {
        match self {
            Self::Frequency => frequency as f64,
            Self::Uniform => 1.0,
            Self::LogScaling => (frequency as f64 + 1.0).ln(),
            Self::PowerScaling(power) => (frequency as f64).powf(*power as f64),
            Self::CapMax(max) => frequency.min(*max) as f64,
            Self::InverseFrequency => 1.0 / (frequency as f64 + 1.0),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum WfcPatternPruningStrategy {
    #[default]
    None,
    FrequencyThreshold(usize),
    FrequencyFraction(f64),
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct WfcLearningStrategy {
    pub periodic_input: bool,
    pub augment_mirror_horizontal: bool,
    pub augment_mirror_vertical: bool,
    pub augment_rotate: bool,
    pub pattern_pruning: WfcPatternPruningStrategy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WfcModel<const N: usize, T: Copy> {
    patterns: Vec<Pattern<N, T>>,
    compatibility: BTreeMap<(PatternId, GridDirection), BitVec>,
    finalized: bool,
}

impl<const N: usize, T: Copy + Eq + Ord> WfcModel<N, T> {
    pub fn new() -> Result<Self, WfcError> {
        if N.is_multiple_of(2) {
            return Err(WfcError::PatternSizeIsNotOdd { provided: N });
        }
        Ok(Self {
            patterns: Default::default(),
            compatibility: Default::default(),
            finalized: false,
        })
    }

    pub fn patterns(&self) -> &[Pattern<N, T>] {
        &self.patterns
    }

    pub fn pattern(&self, id: PatternId) -> Option<&Pattern<N, T>> {
        self.patterns.iter().find(|p| p.id == id)
    }

    pub fn compatibility(&self) -> &BTreeMap<(PatternId, GridDirection), BitVec> {
        &self.compatibility
    }

    pub fn add_pattern(
        &mut self,
        grid: FixedGrid<N, N, T>,
        frequency: usize,
    ) -> Result<PatternId, WfcError> {
        if grid.size().x != N || grid.size().y != N {
            return Err(WfcError::WrongGridSize {
                expected: N,
                provided: grid.size(),
            });
        }
        if let Some(existing) = self
            .patterns
            .iter_mut()
            .find(|p| p.grid.buffer() == grid.buffer())
        {
            existing.frequency += frequency;
            return Ok(existing.id);
        }
        let id = PatternId(self.patterns.len());
        self.patterns.push(Pattern {
            id,
            grid,
            frequency,
            weight: 1.0,
        });
        Ok(id)
    }

    pub fn allow_adjacency(&mut self, from: PatternId, to: PatternId, dir: GridDirection) {
        let entry = self
            .compatibility
            .entry((from, dir))
            .or_insert_with(|| BitVec::repeat(false, self.patterns.len()));
        if to.0 < entry.len() {
            entry.set(to.0, true);
        }
    }

    pub fn finalize(&mut self, weighting_strategy: WfcWeightingStrategy) {
        let pattern_count = self.patterns.len();
        for pattern in &self.patterns {
            for &dir in &[
                GridDirection::North,
                GridDirection::East,
                GridDirection::South,
                GridDirection::West,
            ] {
                self.compatibility
                    .entry((pattern.id, dir))
                    .or_insert_with(|| BitVec::repeat(false, pattern_count));
            }
        }
        for pattern in &mut self.patterns {
            pattern.weight = weighting_strategy.weight(pattern.frequency);
        }
        self.finalized = true;
    }

    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    pub fn learn_overlapping_from_grid(
        &mut self,
        grid: &Grid<T>,
        learning_strategy: WfcLearningStrategy,
    ) -> Result<(), WfcError> {
        if N.is_multiple_of(2) {
            return Err(WfcError::PatternSizeIsNotOdd { provided: N });
        }
        let xmax = if learning_strategy.periodic_input {
            grid.size().x
        } else {
            grid.size().x - N + 1
        };
        let ymax = if learning_strategy.periodic_input {
            grid.size().y
        } else {
            grid.size().y - N + 1
        };

        for y in 0..ymax {
            for x in 0..xmax {
                let pattern = std::array::from_fn(|dy| {
                    std::array::from_fn(|dx| {
                        let xx = (x + dx) % grid.size().x;
                        let yy = (y + dy) % grid.size().y;
                        grid.get((xx, yy)).unwrap()
                    })
                });
                let pattern = FixedGrid::with_buffer(pattern);
                self.add_pattern(pattern, 1)?;

                if learning_strategy.augment_mirror_horizontal {
                    let mirrored_h = pattern.mirrored(true);
                    self.add_pattern(mirrored_h, 1)?;
                }

                if learning_strategy.augment_mirror_vertical {
                    let mirrored_v = pattern.mirrored(false);
                    self.add_pattern(mirrored_v, 1)?;
                }

                if learning_strategy.augment_rotate {
                    let mut pattern = pattern;
                    for _ in 0..3 {
                        pattern = pattern.rotated(true);
                        self.add_pattern(pattern, 1)?;
                    }
                }
            }
        }

        let total_frequency: usize = self.patterns.iter().map(|p| p.frequency).sum();
        match learning_strategy.pattern_pruning {
            WfcPatternPruningStrategy::None => {}
            WfcPatternPruningStrategy::FrequencyThreshold(threshold) => {
                self.patterns.retain(|p| p.frequency >= threshold);
            }
            WfcPatternPruningStrategy::FrequencyFraction(fraction) => {
                let threshold = (total_frequency as f64 * fraction).ceil() as usize;
                self.patterns.retain(|p| p.frequency >= threshold);
            }
        }
        for index in 0..self.patterns.len() {
            self.patterns[index].id = PatternId(index);
        }

        self.rebuild_overlapping_compatibility();
        Ok(())
    }

    pub fn merge(&mut self, other: WfcModel<N, T>) -> Result<(), WfcError> {
        for pattern in other.patterns {
            self.add_pattern(pattern.grid, pattern.frequency)?;
        }

        self.rebuild_overlapping_compatibility();
        Ok(())
    }

    pub fn rebuild_overlapping_compatibility(&mut self) {
        self.compatibility.clear();
        let pattern_count = self.patterns.len();
        for a in 0..pattern_count {
            for b in 0..pattern_count {
                for &dir in &[
                    GridDirection::North,
                    GridDirection::East,
                    GridDirection::South,
                    GridDirection::West,
                ] {
                    if Self::are_patterns_overlap_compatible(
                        &self.patterns[a].grid,
                        &self.patterns[b].grid,
                        dir,
                    ) {
                        self.allow_adjacency(PatternId(a), PatternId(b), dir);
                    }
                }
            }
        }
    }

    fn are_patterns_overlap_compatible(
        a: &FixedGrid<N, N, T>,
        b: &FixedGrid<N, N, T>,
        dir: GridDirection,
    ) -> bool {
        if a.size() != b.size() {
            return false;
        }
        let size = a.size();
        match dir {
            GridDirection::North => {
                let view_a = a.view(Vec2::new(0, 0)..Vec2::new(size.x, size.y - 1));
                let view_b = b.view(Vec2::new(0, 1)..Vec2::new(size.x, size.y));
                for ((_, _, a), (_, _, b)) in view_a.iter().zip(view_b.iter()) {
                    if a != b {
                        return false;
                    }
                }
            }
            GridDirection::South => {
                let view_a = a.view(Vec2::new(0, 1)..Vec2::new(size.x, size.y));
                let view_b = b.view(Vec2::new(0, 0)..Vec2::new(size.x, size.y - 1));
                for ((_, _, a), (_, _, b)) in view_a.iter().zip(view_b.iter()) {
                    if a != b {
                        return false;
                    }
                }
            }
            GridDirection::West => {
                let view_a = a.view(Vec2::new(0, 0)..Vec2::new(size.x - 1, size.y));
                let view_b = b.view(Vec2::new(1, 0)..Vec2::new(size.x, size.y));
                for ((_, _, a), (_, _, b)) in view_a.iter().zip(view_b.iter()) {
                    if a != b {
                        return false;
                    }
                }
            }
            GridDirection::East => {
                let view_a = a.view(Vec2::new(1, 0)..Vec2::new(size.x, size.y));
                let view_b = b.view(Vec2::new(0, 0)..Vec2::new(size.x - 1, size.y));
                for ((_, _, a), (_, _, b)) in view_a.iter().zip(view_b.iter()) {
                    if a != b {
                        return false;
                    }
                }
            }
            _ => unreachable!(),
        }
        true
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum WfcCollapseResult<T: Copy> {
    Incomplete,
    Complete { grid: Grid<T> },
    Impossible,
    ModelNotFinalized,
}

pub struct WfcCellModifier<'a, const N: usize, T: Copy> {
    solver: &'a mut WfcSolver,
    index: usize,
    model: &'a WfcModel<N, T>,
}

impl<'a, const N: usize, T: Copy> Drop for WfcCellModifier<'a, N, T> {
    fn drop(&mut self) {
        self.solver.propagate(self.model);
    }
}

impl<'a, const N: usize, T: Copy> WfcCellModifier<'a, N, T> {
    pub fn clear(&mut self) -> &mut Self {
        self.solver.possibility_space[self.index].fill(false);
        self
    }

    pub fn all(&mut self) -> &mut Self {
        self.solver.possibility_space[self.index].fill(true);
        self
    }

    pub fn allow(&mut self, pattern: PatternId) -> &mut Self {
        let cell = &mut self.solver.possibility_space[self.index];
        if pattern.0 < cell.len() {
            cell.set(pattern.0, true);
        }
        self
    }

    pub fn forbid(&mut self, pattern: PatternId) -> &mut Self {
        let cell = &mut self.solver.possibility_space[self.index];
        if pattern.0 < cell.len() {
            cell.set(pattern.0, false);
        }
        self
    }
}

pub struct WfcSolver {
    size: Vec2<usize>,
    possibility_space: Vec<BitVec>,
    iterations: usize,
    pub desired_starting_location: Option<Vec2<usize>>,
}

impl WfcSolver {
    pub fn new<const N: usize, T: Copy>(
        size: impl Into<Vec2<usize>>,
        model: &WfcModel<N, T>,
    ) -> Self {
        let size = size.into();
        Self {
            size,
            possibility_space: vec![BitVec::repeat(true, model.patterns.len()); size.x * size.y],
            iterations: 0,
            desired_starting_location: None,
        }
    }

    pub fn with_desired_starting_location(mut self, loc: impl Into<Vec2<usize>>) -> Self {
        self.desired_starting_location = Some(loc.into());
        self
    }

    pub fn modify_cell<'a, const N: usize, T: Copy>(
        &'a mut self,
        model: &'a WfcModel<N, T>,
        location: impl Into<Vec2<usize>>,
    ) -> WfcCellModifier<'a, N, T> {
        let location = location.into();
        let index = self.index(location.x, location.y);
        WfcCellModifier {
            solver: self,
            index,
            model,
        }
    }

    pub fn set_cell<const N: usize, T: Copy>(
        &mut self,
        model: &WfcModel<N, T>,
        location: impl Into<Vec2<usize>>,
        allowed: &[PatternId],
    ) {
        let location = location.into();
        let idx = self.index(location.x, location.y);
        self.possibility_space[idx].fill(false);
        for p in allowed {
            self.possibility_space[idx].set(p.0, true);
        }
        self.propagate(model);
    }

    pub fn size(&self) -> Vec2<usize> {
        self.size
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the current and total number of possible patterns across all cells.
    pub fn uncertainty(&self) -> (usize, usize) {
        let current = self
            .possibility_space
            .iter()
            .map(|bv| bv.count_ones().saturating_sub(1))
            .sum();
        let total =
            self.possibility_space.len() * self.possibility_space[0].len().saturating_sub(1);
        (current, total)
    }

    pub fn collapse_step<const N: usize, T: Copy>(
        &mut self,
        model: &WfcModel<N, T>,
        rng: &mut impl Rng,
    ) -> WfcCollapseResult<T> {
        if !model.finalized {
            return WfcCollapseResult::ModelNotFinalized;
        }
        self.iterations += 1;

        let mut best_entropy = f64::INFINITY;
        let mut choice: Option<(usize, usize)> = None;

        if let Some(loc) = self.desired_starting_location.take()
            && loc.x < self.size.x
            && loc.y < self.size.y
        {
            let idx = self.index(loc.x, loc.y);
            let possibilities = &self.possibility_space[idx];
            if possibilities.count_ones() > 1 {
                choice = Some((loc.x, loc.y));
            }
        }

        if choice.is_none() {
            for y in 0..self.size.y {
                for x in 0..self.size.x {
                    let idx = self.index(x, y);
                    let possibilities = &self.possibility_space[idx];

                    let count = possibilities.count_ones();
                    if count <= 1 {
                        continue;
                    }

                    let mut sum_w = 0.0;
                    let mut sum_w_log = 0.0;
                    for p in 0..model.patterns.len() {
                        if possibilities[p] {
                            let w = model.patterns[p].weight;
                            sum_w += w;
                            sum_w_log += w * w.ln();
                        }
                    }

                    if sum_w > 0.0 {
                        let entropy = sum_w.ln() - (sum_w_log / sum_w);
                        let entropy = entropy + rng.random_range(0.0..1e-6);
                        if entropy < best_entropy {
                            best_entropy = entropy;
                            choice = Some((x, y));
                        }
                    }
                }
            }
        }

        if let Some((x, y)) = choice {
            let idx = self.index(x, y);
            let mut candidates = Vec::new();
            for p in 0..model.patterns.len() {
                if self.possibility_space[idx][p] {
                    candidates.push((p, model.patterns[p].weight));
                }
            }
            if candidates.is_empty() {
                return WfcCollapseResult::Impossible;
            }

            let total_weight: f64 = candidates.iter().map(|(_, w)| *w).sum();
            let mut roll = rng.random_range(0.0..total_weight);
            for (pattern_index, w) in candidates {
                roll -= w;
                if roll <= 0.0 {
                    self.possibility_space[idx].fill(false);
                    self.possibility_space[idx].set(pattern_index, true);
                    break;
                }
            }

            self.propagate(model);
            WfcCollapseResult::Incomplete
        } else {
            for cell in &self.possibility_space {
                if cell.not_any() {
                    return WfcCollapseResult::Impossible;
                }
            }

            let mut grid = Grid::new(self.size, model.patterns[0].grid.get((0, 0)).unwrap());
            for y in 0..self.size.y {
                for x in 0..self.size.x {
                    let idx = self.index(x, y);
                    let p = self.possibility_space[idx]
                        .iter()
                        .position(|b| *b)
                        .expect("cell should have collapsed");
                    let offset = Vec2::new(N / 2, N / 2);
                    grid.set((x, y), model.patterns[p].grid.get(offset).unwrap());
                }
            }
            WfcCollapseResult::Complete { grid }
        }
    }

    pub fn collapse<const N: usize, T: Copy>(
        &mut self,
        model: &WfcModel<N, T>,
        rng: &mut impl Rng,
    ) -> WfcCollapseResult<T> {
        loop {
            match self.collapse_step(model, rng) {
                WfcCollapseResult::Incomplete => continue,
                result => return result,
            }
        }
    }

    pub async fn collapse_async<const N: usize, T: Copy>(
        &mut self,
        model: &WfcModel<N, T>,
        rng: &mut impl Rng,
    ) -> WfcCollapseResult<T> {
        std::future::poll_fn(|cx| match self.collapse_step(model, rng) {
            WfcCollapseResult::Incomplete => {
                cx.waker().wake_by_ref();
                std::task::Poll::Pending
            }
            result => std::task::Poll::Ready(result),
        })
        .await
    }

    fn propagate<const N: usize, T: Copy>(&mut self, model: &WfcModel<N, T>) {
        let mut stack: Vec<(usize, usize)> = (0..self.size.y)
            .flat_map(|y| (0..self.size.x).map(move |x| (x, y)))
            .collect();

        while let Some((x, y)) = stack.pop() {
            let idx = self.index(x, y);
            for &dir in &[
                GridDirection::North,
                GridDirection::East,
                GridDirection::South,
                GridDirection::West,
            ] {
                let (nx, ny) = match dir {
                    GridDirection::North if y > 0 => (x, y - 1),
                    GridDirection::South if y + 1 < self.size.y => (x, y + 1),
                    GridDirection::West if x > 0 => (x - 1, y),
                    GridDirection::East if x + 1 < self.size.x => (x + 1, y),
                    _ => continue,
                };
                let nidx = self.index(nx, ny);

                let mut allowed = BitVec::repeat(false, model.patterns.len());
                for a in 0..model.patterns.len() {
                    if self.possibility_space[idx][a]
                        && let Some(compat) = model.compatibility.get(&(PatternId(a), dir))
                    {
                        allowed |= compat.clone();
                    }
                }

                let before = self.possibility_space[nidx].clone();
                self.possibility_space[nidx] &= allowed;

                if self.possibility_space[nidx] != before {
                    stack.push((nx, ny));
                }
            }
        }
    }

    fn index(&self, x: usize, y: usize) -> usize {
        y * self.size.x + x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, Pixel, RgbaImage};
    use rand::SeedableRng;

    #[test]
    #[cfg(feature = "diagnostics")]
    fn test_model_overlapping() {
        let grid = Grid::with_buffer(
            (5, 5),
            vec![
                0, 0, 0, 0, 0, //
                0, 0, 0, 0, 0, //
                0, 0, 1, 0, 0, //
                0, 0, 0, 0, 0, //
                0, 0, 0, 0, 0, //
            ],
        )
        .unwrap();

        let mut model = WfcModel::<3, _>::new().unwrap();
        model
            .learn_overlapping_from_grid(&grid, WfcLearningStrategy::default())
            .unwrap();
        model.finalize(WfcWeightingStrategy::Frequency);
        std::fs::write(
            "resources/wfc-model-overlap.html",
            model.diagnostics().unwrap(),
        )
        .unwrap();

        let mut solver = WfcSolver::new((5, 5), &model);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        solver.set_cell(&model, (2, 2), &[PatternId(4)]);

        match solver.collapse(&model, &mut rng) {
            WfcCollapseResult::Complete { grid } => {
                assert_eq!(solver.iterations(), 3);
                assert_eq!(
                    grid.buffer(),
                    vec![
                        0, 0, 0, 0, 0, //
                        0, 0, 0, 0, 0, //
                        0, 0, 1, 0, 0, //
                        0, 0, 0, 0, 0, //
                        0, 0, 0, 0, 0, //
                    ]
                );
            }
            WfcCollapseResult::Impossible => {
                panic!("WFC reported impossible for a solvable grid");
            }
            WfcCollapseResult::Incomplete => {
                panic!("WFC reported incomplete after solve");
            }
            WfcCollapseResult::ModelNotFinalized => {
                panic!("WFC model was not finalized");
            }
        }
    }

    #[test]
    fn test_wfc_learn_overlapping() {
        const OUTPUT_SIZE: Vec2<usize> = Vec2::new(40, 20);

        println!("Load input image...");
        let img =
            image::open("resources/wfc-input-overlap.png").expect("Failed to open input image");
        let (width, height) = img.dimensions();

        println!("Turn input pixels into image grid...");
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let p = img.get_pixel(x, y);
                let p = p.channels();
                pixels.push([p[0], p[1], p[2]]);
            }
        }
        let grid = Grid::with_buffer((width as usize, height as usize), pixels).unwrap();

        println!("Learn patterns from image grid...");
        let mut model = WfcModel::<3, _>::new().unwrap();
        model
            .learn_overlapping_from_grid(
                &grid,
                WfcLearningStrategy {
                    augment_mirror_horizontal: true,
                    ..Default::default()
                },
            )
            .unwrap();
        model.finalize(WfcWeightingStrategy::Frequency);
        println!("Learned {} patterns", model.patterns().len());

        {
            let mut freq_map = BTreeMap::new();
            for p in model.patterns() {
                *freq_map.entry(p.frequency()).or_insert(0) += 1;
            }
            println!("Pattern frequency distribution:");
            for (freq, count) in freq_map {
                println!("  Frequency {}: {} patterns", freq, count);
            }
        }

        println!("Solve WFC to generate output grid...");
        let mut solver = WfcSolver::new(OUTPUT_SIZE, &model)
            .with_desired_starting_location((OUTPUT_SIZE.x / 2, OUTPUT_SIZE.y - 1));

        let ground = model
            .patterns()
            .iter()
            .find(|p| {
                let center = p.grid.get((1, 1)).unwrap();
                let bottom = p.grid.get((1, 2)).unwrap();
                center[0] == 0
                    && center[1] == 170
                    && center[2] == 0
                    && bottom[0] == 185
                    && bottom[1] == 122
                    && bottom[2] == 87
            })
            .map(|p| p.id)
            .expect("No ground pattern found!");
        solver.set_cell(&model, solver.desired_starting_location.unwrap(), &[ground]);

        let mut rng = rand::rng();
        let result = loop {
            let (current, total) = solver.uncertainty();
            match solver.collapse_step(&model, &mut rng) {
                WfcCollapseResult::Incomplete => {
                    println!(
                        "WFC collapse | iterations: {} | uncertainty: {}/{}",
                        solver.iterations(),
                        current,
                        total
                    );
                    continue;
                }
                result => {
                    println!(
                        "WFC collapsed | iterations: {} | uncertainty: {}/{}",
                        solver.iterations(),
                        current,
                        total
                    );
                    break result;
                }
            }
        };

        println!("Handle WFC result...");
        match result {
            WfcCollapseResult::Complete { grid, .. } => {
                let mut out_img = RgbaImage::new(OUTPUT_SIZE.x as u32, OUTPUT_SIZE.y as u32);
                for y in 0..OUTPUT_SIZE.y {
                    for x in 0..OUTPUT_SIZE.x {
                        let pixel = grid.get((x, y)).unwrap();
                        let r = pixel[0];
                        let g = pixel[1];
                        let b = pixel[2];
                        out_img.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, 255]));
                    }
                }
                out_img
                    .save("resources/wfc-output-overlap.png")
                    .expect("Failed to save output image");
            }
            WfcCollapseResult::Impossible => {
                panic!("WFC reported impossible for image input");
            }
            WfcCollapseResult::Incomplete => {
                panic!("WFC reported incomplete after solve");
            }
            WfcCollapseResult::ModelNotFinalized => {
                panic!("WFC model was not finalized");
            }
        }
    }
}
