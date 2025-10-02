pub mod auto_tile;

use crate::grid::Grid;
use noise::NoiseFn;
use std::ops::{Add, Div, Mul, Range, Sub};
use vek::{Mat4, Vec2};

pub trait GridGenetator<T: Copy> {
    fn generate(
        &mut self,
        location: Vec2<usize>,
        size: Vec2<usize>,
        current: T,
        grid: &Grid<T>,
    ) -> T;
}

impl<T: Copy, F: FnMut(Vec2<usize>, Vec2<usize>, T) -> T> GridGenetator<T> for F {
    fn generate(&mut self, location: Vec2<usize>, size: Vec2<usize>, current: T, _: &Grid<T>) -> T {
        self(location, size, current)
    }
}

pub struct ConstGenerator<T: Copy>(pub T);

impl<T: Copy> GridGenetator<T> for ConstGenerator<T> {
    fn generate(&mut self, _: Vec2<usize>, _: Vec2<usize>, _: T, _: &Grid<T>) -> T {
        self.0
    }
}

pub struct OffsetLocationGenerator<'a, T: Copy> {
    pub generator: &'a mut dyn GridGenetator<T>,
    pub offsets: &'a Grid<Vec2<isize>>,
}

impl<T: Copy> GridGenetator<T> for OffsetLocationGenerator<'_, T> {
    fn generate(
        &mut self,
        mut location: Vec2<usize>,
        size: Vec2<usize>,
        current: T,
        grid: &Grid<T>,
    ) -> T {
        let offset = self.offsets.get(location).unwrap_or_default();
        if offset.x >= 0 {
            location.x = (location.x + offset.x as usize) % size.x;
        } else {
            location.x = (location.x + size.x - offset.x.unsigned_abs() % size.x) % size.x;
        }
        if offset.y >= 0 {
            location.y = (location.y + offset.y as usize) % size.y;
        } else {
            location.y = (location.y + size.y - offset.y.unsigned_abs() % size.y) % size.y;
        }
        self.generator.generate(location, size, current, grid)
    }
}

pub struct NoiseGenerator<T: NoiseFn<f64, 2>> {
    pub noise: T,
    pub transform: Mat4<f64>,
}

impl<T: NoiseFn<f64, 2>> NoiseGenerator<T> {
    pub fn new(noise: T) -> Self {
        Self {
            noise,
            transform: Mat4::identity(),
        }
    }

    pub fn transform(mut self, transform: Mat4<f64>) -> Self {
        self.transform = transform;
        self
    }
}

impl<T: NoiseFn<f64, 2>> GridGenetator<f64> for NoiseGenerator<T> {
    fn generate(&mut self, location: Vec2<usize>, _: Vec2<usize>, _: f64, _: &Grid<f64>) -> f64 {
        let point = self.transform.mul_point(Vec2 {
            x: location.x as f64,
            y: location.y as f64,
        });
        self.noise.get(point.into_array())
    }
}

pub struct CopyGenerator<'a, T: Copy> {
    pub other: &'a Grid<T>,
}

impl<T: Copy + Add<Output = T> + Default> GridGenetator<T> for CopyGenerator<'_, T> {
    fn generate(&mut self, location: Vec2<usize>, _: Vec2<usize>, _: T, _: &Grid<T>) -> T {
        self.other.get(location).unwrap_or_default()
    }
}

pub struct AddGenerator<'a, T: Copy> {
    pub other: &'a Grid<T>,
}

impl<T: Copy + Add<Output = T> + Default> GridGenetator<T> for AddGenerator<'_, T> {
    fn generate(&mut self, location: Vec2<usize>, _: Vec2<usize>, current: T, _: &Grid<T>) -> T {
        current + self.other.get(location).unwrap_or_default()
    }
}

pub struct SubGenerator<'a, T: Copy> {
    pub other: &'a Grid<T>,
}

impl<T: Copy + Sub<Output = T> + Default> GridGenetator<T> for SubGenerator<'_, T> {
    fn generate(&mut self, location: Vec2<usize>, _: Vec2<usize>, current: T, _: &Grid<T>) -> T {
        current - self.other.get(location).unwrap_or_default()
    }
}

pub struct MulGenerator<'a, T: Copy> {
    pub other: &'a Grid<T>,
}

impl<T: Copy + Mul<Output = T> + Default> GridGenetator<T> for MulGenerator<'_, T> {
    fn generate(&mut self, location: Vec2<usize>, _: Vec2<usize>, current: T, _: &Grid<T>) -> T {
        current * self.other.get(location).unwrap_or_default()
    }
}

pub struct DivGenerator<'a, T: Copy> {
    pub other: &'a Grid<T>,
}

impl<T: Copy + Div<Output = T> + Default> GridGenetator<T> for DivGenerator<'_, T> {
    fn generate(&mut self, location: Vec2<usize>, _: Vec2<usize>, current: T, _: &Grid<T>) -> T {
        current / self.other.get(location).unwrap_or_default()
    }
}

pub struct MinGenerator<'a, T: Copy> {
    pub other: &'a Grid<T>,
}

impl<T: Copy + Div<Output = T> + Ord + Default> GridGenetator<T> for MinGenerator<'_, T> {
    fn generate(&mut self, location: Vec2<usize>, _: Vec2<usize>, current: T, _: &Grid<T>) -> T {
        current.min(self.other.get(location).unwrap_or_default())
    }
}

pub struct MaxGenerator<'a, T: Copy> {
    pub other: &'a Grid<T>,
}

impl<T: Copy + Div<Output = T> + Ord + Default> GridGenetator<T> for MaxGenerator<'_, T> {
    fn generate(&mut self, location: Vec2<usize>, _: Vec2<usize>, current: T, _: &Grid<T>) -> T {
        current.max(self.other.get(location).unwrap_or_default())
    }
}

pub struct ClampGenerator<T: Copy> {
    pub min: T,
    pub max: T,
}

impl<T: Copy + Ord + Default> GridGenetator<T> for ClampGenerator<T> {
    fn generate(&mut self, _: Vec2<usize>, _: Vec2<usize>, current: T, _: &Grid<T>) -> T {
        current.clamp(self.min, self.max)
    }
}

pub struct RemapGenerator<T: Copy> {
    pub from: Range<T>,
    pub to: Range<T>,
}

impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>>
    GridGenetator<T> for RemapGenerator<T>
{
    fn generate(&mut self, _: Vec2<usize>, _: Vec2<usize>, current: T, _: &Grid<T>) -> T {
        let factor = (current - self.from.start) / (self.from.end - self.from.start);
        (self.to.end - self.to.start) * factor + self.to.start
    }
}

pub enum ThresholdGenerator<'a, T: Copy> {
    Constant {
        threshold: T,
        value_upper: T,
        value_lower: T,
    },
    Samples {
        thresholds: &'a Grid<T>,
        value_upper: T,
        value_lower: T,
    },
}

impl<T: Copy + PartialOrd + Default> GridGenetator<T> for ThresholdGenerator<'_, T> {
    fn generate(&mut self, location: Vec2<usize>, _: Vec2<usize>, current: T, _: &Grid<T>) -> T {
        match self {
            Self::Constant {
                threshold,
                value_upper,
                value_lower,
            } => {
                if current > *threshold {
                    *value_upper
                } else {
                    *value_lower
                }
            }
            Self::Samples {
                thresholds,
                value_upper,
                value_lower,
            } => {
                if current > thresholds.get(location).unwrap_or_default() {
                    *value_upper
                } else {
                    *value_lower
                }
            }
        }
    }
}

pub struct Kernel33Generator<'a, T: Copy> {
    pub other: &'a Grid<T>,
    pub kernel: [T; 9],
}

impl<'a> Kernel33Generator<'a, f64> {
    pub fn identity(other: &'a Grid<f64>) -> Self {
        Self {
            other,
            kernel: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn ridge(other: &'a Grid<f64>) -> Self {
        Self {
            other,
            kernel: [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
        }
    }

    pub fn edge_detection(other: &'a Grid<f64>) -> Self {
        Self {
            other,
            kernel: [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
        }
    }

    pub fn sharpen(other: &'a Grid<f64>) -> Self {
        Self {
            other,
            kernel: [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
        }
    }

    pub fn emboss(other: &'a Grid<f64>) -> Self {
        Self {
            other,
            kernel: [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0],
        }
    }

    pub fn box_blur(other: &'a Grid<f64>) -> Self {
        Self {
            other,
            kernel: [
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
            ],
        }
    }

    pub fn gaussian_blur(other: &'a Grid<f64>) -> Self {
        Self {
            other,
            kernel: [
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                4.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
            ],
        }
    }
}

impl<T: Copy + Add<Output = T> + Mul<Output = T> + Default> GridGenetator<T>
    for Kernel33Generator<'_, T>
{
    fn generate(&mut self, location: Vec2<usize>, size: Vec2<usize>, _: T, _: &Grid<T>) -> T {
        let region = [
            self.other
                .get(location + Vec2::new(size.x - 1, size.y - 1))
                .unwrap_or_default(),
            self.other
                .get(location + Vec2::new(0, size.y - 1))
                .unwrap_or_default(),
            self.other
                .get(location + Vec2::new(1, size.y - 1))
                .unwrap_or_default(),
            self.other
                .get(location + Vec2::new(size.x - 1, 0))
                .unwrap_or_default(),
            self.other.get(location).unwrap_or_default(),
            self.other
                .get(location + Vec2::new(1, 0))
                .unwrap_or_default(),
            self.other
                .get(location + Vec2::new(size.x - 1, 1))
                .unwrap_or_default(),
            self.other
                .get(location + Vec2::new(0, 1))
                .unwrap_or_default(),
            self.other
                .get(location + Vec2::new(1, 1))
                .unwrap_or_default(),
        ];
        region
            .into_iter()
            .zip(self.kernel)
            .fold(Default::default(), |accumulator, (value, kernel)| {
                value * kernel + accumulator
            })
    }
}
