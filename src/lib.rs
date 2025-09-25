pub mod generators;
pub mod grid;
pub mod wfc;

pub mod third_party {
    pub use noise;
    pub use serde;
    pub use vek;
}

#[cfg(test)]
mod tests {
    use crate::{
        generators::{
            GridGenetator, Kernel33Generator, NoiseGenerator, OffsetLocationGenerator,
            RemapGenerator, SubGenerator, ThresholdGenerator,
        },
        grid::{Grid, GridDirection},
    };
    use image::{GrayImage, RgbImage};
    use noise::{Fbm, MultiFractal, ScalePoint, SuperSimplex, Worley};
    use serde::{Deserialize, Serialize};
    use vek::Vec2;

    const SIZE: usize = 512;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Terrain {
        Water,
        Sand,
        Grass,
        Mountain,
    }

    fn gradient_generator(location: Vec2<usize>, size: Vec2<usize>, _: f64) -> f64 {
        let center = size / 2;
        let x = location.x.abs_diff(center.x) as f64;
        let y = location.y.abs_diff(center.y) as f64;
        let result = (x / center.x as f64).max(y / center.y as f64);
        result * result
    }

    struct OffsetsGenerator<'a> {
        pub source: &'a Grid<f64>,
        pub scale: f64,
    }

    impl GridGenetator<Vec2<isize>> for OffsetsGenerator<'_> {
        fn generate(
            &mut self,
            location: Vec2<usize>,
            _: Vec2<usize>,
            _: Vec2<isize>,
            _: &Grid<Vec2<isize>>,
        ) -> Vec2<isize> {
            let left = self
                .source
                .get(
                    self.source
                        .location_offset(location, GridDirection::West, 1)
                        .unwrap_or(location),
                )
                .unwrap_or_default();
            let right = self
                .source
                .get(
                    self.source
                        .location_offset(location, GridDirection::East, 1)
                        .unwrap_or(location),
                )
                .unwrap_or_default();
            let top = self
                .source
                .get(
                    self.source
                        .location_offset(location, GridDirection::North, 1)
                        .unwrap_or(location),
                )
                .unwrap_or_default();
            let bottom = self
                .source
                .get(
                    self.source
                        .location_offset(location, GridDirection::South, 1)
                        .unwrap_or(location),
                )
                .unwrap_or_default();
            Vec2 {
                x: ((right - left) * self.scale) as isize,
                y: ((bottom - top) * self.scale) as isize,
            }
        }
    }

    fn generate_terrain(size: Vec2<usize>) -> Grid<Terrain> {
        let mut grid = Grid::<f64>::generate(
            size,
            NoiseGenerator::new(
                Fbm::<SuperSimplex>::default()
                    .set_octaves(9)
                    .set_frequency(0.008),
            ),
        );
        grid.apply_all(RemapGenerator {
            from: -1.0..1.0,
            to: 0.0..1.0,
        });
        let gradient = grid.fork_generate(gradient_generator);
        grid.apply_all(SubGenerator { other: &gradient });
        grid.map(|_, _, value| {
            if value > 0.5 {
                Terrain::Mountain
            } else if value > 0.2 {
                Terrain::Grass
            } else if value > 0.15 {
                Terrain::Sand
            } else {
                Terrain::Water
            }
        })
    }

    fn generate_tunnels(size: Vec2<usize>) -> Grid<bool> {
        let offsets = Grid::<f64>::generate(
            size,
            NoiseGenerator::new(ScalePoint::new(SuperSimplex::default()).set_scale(0.04)),
        );
        let offsets = Grid::<Vec2<isize>>::generate(
            offsets.size(),
            OffsetsGenerator {
                source: &offsets,
                scale: 20.0,
            },
        );
        let mut thresholds = Grid::<f64>::generate(
            size,
            NoiseGenerator::new(ScalePoint::new(SuperSimplex::default()).set_scale(0.02)),
        );
        thresholds.apply_all(RemapGenerator {
            from: -1.0..1.0,
            to: 0.0..0.4,
        });
        let mut grid = Grid::<f64>::generate(
            size,
            OffsetLocationGenerator {
                generator: &mut NoiseGenerator::new(Worley::default().set_frequency(0.04)),
                offsets: &offsets,
            },
        );
        grid.apply_all(RemapGenerator {
            from: -1.0..1.0,
            to: 0.0..1.0,
        });
        grid.apply_all(Kernel33Generator::edge_detection(&grid.clone()));
        grid.apply_all(ThresholdGenerator::Constant {
            threshold: 1.0e-4,
            value_upper: 1.0,
            value_lower: 0.0,
        });
        for _ in 0..1 {
            grid.apply_all(Kernel33Generator::gaussian_blur(&grid.clone()));
        }
        grid.apply_all(ThresholdGenerator::Samples {
            thresholds: &thresholds,
            value_upper: 1.0,
            value_lower: 0.0,
        });
        grid.map(|_, _, value| value >= 0.5)
    }

    #[test]
    fn test_pcg_island() {
        let terrain = generate_terrain(SIZE.into());
        let tunnels = generate_tunnels(SIZE.into());

        let (size, buffer) = terrain.into_inner();
        let buffer = buffer
            .into_iter()
            .enumerate()
            .flat_map(|(index, value)| match value {
                Terrain::Mountain => {
                    let location = tunnels.location(index);
                    if tunnels.get(location).unwrap() {
                        [64, 64, 64]
                    } else {
                        [128, 128, 128]
                    }
                }
                Terrain::Grass => [0, 128, 0],
                Terrain::Sand => [192, 192, 128],
                Terrain::Water => [0, 0, 128],
            })
            .collect();
        let image = RgbImage::from_vec(size.x as _, size.y as _, buffer).unwrap();
        image.save("./resources/island.png").unwrap();
    }

    #[test]
    fn test_pcg_tunnels() {
        let tunnels = generate_tunnels(SIZE.into());

        let (size, buffer) = tunnels.into_inner();
        let buffer = buffer
            .into_iter()
            .map(|value| if value { 255 } else { 0 })
            .collect();
        let image = GrayImage::from_vec(size.x as _, size.y as _, buffer).unwrap();
        image.save("./resources/caves.png").unwrap();
    }

    #[test]
    fn test_serde() {
        fn is_serde<T: Serialize + for<'d> Deserialize<'d>>() {}

        is_serde::<Grid<usize>>();
        is_serde::<Grid<f64>>();

        let grid = Grid::new(Vec2::new(10, 10), 0usize);
        let serialized = serde_json::to_string(&grid).unwrap();
        let deserialized: Grid<usize> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(grid.size(), deserialized.size());
        assert_eq!(grid.buffer(), deserialized.buffer());

        let grid = Grid::new(Vec2::new(10, 10), 0.0f64);
        let serialized = serde_json::to_string(&grid).unwrap();
        let deserialized: Grid<f64> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(grid.size(), deserialized.size());
        assert_eq!(grid.buffer(), deserialized.buffer());
    }
}
