use crate::{
    generators::GridGenetator,
    grid::{FixedGrid, Grid},
};
use rand::Rng;
use vek::Vec2;

#[derive(Debug)]
pub struct AutoTileRule<const N: usize, S: Copy + PartialEq, O: Copy> {
    pub pattern: FixedGrid<N, N, Option<S>>,
    pub tiles: Vec<O>,
    pub probability: f32,
    pub stop_on_match: bool,
}

impl<const N: usize, S: Copy + PartialEq, O: Copy> AutoTileRule<N, S, O> {
    pub fn new(pattern: FixedGrid<N, N, Option<S>>, tiles: impl IntoIterator<Item = O>) -> Self {
        Self {
            pattern,
            tiles: tiles.into_iter().collect(),
            probability: 1.0,
            stop_on_match: true,
        }
    }

    pub fn tile(mut self, tile: O) -> Self {
        self.tiles.push(tile);
        self
    }

    pub fn tiles(mut self, tiles: impl IntoIterator<Item = O>) -> Self {
        self.tiles.extend(tiles);
        self
    }

    pub fn probability(mut self, probability: f32) -> Self {
        self.probability = probability;
        self
    }

    pub fn stop_on_match(mut self, stop_on_match: bool) -> Self {
        self.stop_on_match = stop_on_match;
        self
    }

    pub fn does_match(&self, grid: &Grid<S>, location: Vec2<usize>) -> bool {
        for y_pattern in 0..N {
            let y_grid = location.y as isize + y_pattern as isize - (N as isize / 2);
            for x_pattern in 0..N {
                let x_grid = location.x as isize + x_pattern as isize - (N as isize / 2);
                let pattern_cell = self.pattern.get((x_pattern, y_pattern)).flatten();
                if let Some(pattern_cell) = pattern_cell {
                    let grid_cell = if y_grid < 0 || x_grid < 0 {
                        None
                    } else {
                        grid.get((x_grid as usize, y_grid as usize))
                    };
                    if Some(pattern_cell) != grid_cell {
                        return false;
                    }
                }
            }
        }
        true
    }
}

#[derive(Debug)]
pub struct AutoTileModel<const N: usize, S: Copy + PartialEq, O: Copy> {
    pub rules: Vec<AutoTileRule<N, S, O>>,
    pub default_tile: O,
}

impl<const N: usize, S: Copy + PartialEq, O: Copy> Default for AutoTileModel<N, S, O>
where
    O: Default,
{
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            default_tile: O::default(),
        }
    }
}

impl<const N: usize, S: Copy + PartialEq, O: Copy> AutoTileModel<N, S, O> {
    pub fn new(rules: impl IntoIterator<Item = AutoTileRule<N, S, O>>, default_tile: O) -> Self {
        Self {
            rules: rules.into_iter().collect(),
            default_tile,
        }
    }

    pub fn rule(mut self, rule: AutoTileRule<N, S, O>) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn rules(mut self, rules: impl IntoIterator<Item = AutoTileRule<N, S, O>>) -> Self {
        self.rules.extend(rules);
        self
    }

    pub fn default_tile(mut self, default_tile: O) -> Self {
        self.default_tile = default_tile;
        self
    }

    pub fn get_matching_rules<'a>(
        &'a self,
        grid: &Grid<S>,
        location: Vec2<usize>,
        output: &mut Vec<&'a AutoTileRule<N, S, O>>,
    ) {
        for rule in &self.rules {
            if rule.does_match(grid, location) {
                output.push(rule);
                if rule.stop_on_match {
                    return;
                }
            }
        }
    }

    pub fn select_tile<'a>(
        &'a self,
        grid: &Grid<S>,
        location: Vec2<usize>,
        rng: &mut impl Rng,
        matching_rules_cache: &mut Vec<&'a AutoTileRule<N, S, O>>,
    ) -> O {
        matching_rules_cache.clear();
        self.get_matching_rules(grid, location, matching_rules_cache);
        if matching_rules_cache.is_empty() {
            return self.default_tile;
        }
        let total_probability = matching_rules_cache
            .iter()
            .map(|r| r.probability)
            .sum::<f32>();
        let mut random_value = rng.random_range(0.0..total_probability);
        for rule in matching_rules_cache {
            if random_value < rule.probability {
                let tile_index = rng.random_range(0..rule.tiles.len());
                return rule.tiles[tile_index];
            }
            random_value -= rule.probability;
        }
        self.default_tile
    }
}

pub struct AutoTilingGenerator<'a, const N: usize, S: Copy + PartialEq, O: Copy, R: Rng> {
    pub model: &'a AutoTileModel<N, S, O>,
    pub source: &'a Grid<S>,
    pub rng: R,
    matching_rules_cache: Vec<&'a AutoTileRule<N, S, O>>,
}

impl<'a, const N: usize, S: Copy + PartialEq, O: Copy, R: Rng> AutoTilingGenerator<'a, N, S, O, R> {
    pub fn new(model: &'a AutoTileModel<N, S, O>, source: &'a Grid<S>, rng: R) -> Self {
        Self {
            model,
            source,
            rng,
            matching_rules_cache: Vec::new(),
        }
    }
}

impl<const N: usize, S: Copy + PartialEq, O: Copy, R: Rng> GridGenetator<O>
    for AutoTilingGenerator<'_, N, S, O, R>
{
    fn generate(&mut self, location: Vec2<usize>, _: Vec2<usize>, _: O, _: &Grid<O>) -> O {
        self.model.select_tile(
            self.source,
            location,
            &mut self.rng,
            &mut self.matching_rules_cache,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_tile_rule_does_match() {
        let rule = AutoTileRule::new(
            FixedGrid::<3, 3, Option<u8>>::with_buffer([
                [None, Some(1), None],
                [Some(1), None, Some(1)],
                [None, Some(1), None],
            ]),
            [42],
        );

        let grid = Grid::with_buffer(
            (3, 3),
            vec![
                0, 1, 0, //
                1, 2, 1, //
                0, 1, 0, //
            ],
        )
        .unwrap();

        assert!(!rule.does_match(&grid, Vec2::new(0, 0)));
        assert!(rule.does_match(&grid, Vec2::new(1, 1)));
        assert!(!rule.does_match(&grid, Vec2::new(2, 2)));
    }

    #[test]
    fn test_auto_tiling() {
        let model = AutoTileModel::new(
            [
                AutoTileRule::new(
                    FixedGrid::<3, 3, Option<u8>>::with_buffer([
                        [None, None, None],
                        [None, Some(0), Some(1)],
                        [None, Some(1), None],
                    ]),
                    [1],
                ),
                AutoTileRule::new(
                    FixedGrid::<3, 3, Option<u8>>::with_buffer([
                        [None, None, None],
                        [Some(0), Some(1), None],
                        [None, Some(0), None],
                    ]),
                    [2],
                ),
                AutoTileRule::new(
                    FixedGrid::<3, 3, Option<u8>>::with_buffer([
                        [None, Some(0), None],
                        [None, Some(1), Some(0)],
                        [None, None, None],
                    ]),
                    [3],
                ),
                AutoTileRule::new(
                    FixedGrid::<3, 3, Option<u8>>::with_buffer([
                        [None, Some(1), None],
                        [Some(1), Some(0), None],
                        [None, None, None],
                    ]),
                    [4],
                ),
            ],
            0,
        );

        let source = Grid::with_buffer(
            (2, 2),
            vec![
                0, 1, //
                1, 0, //
            ],
        )
        .unwrap();

        let output = Grid::generate(
            (2, 2),
            AutoTilingGenerator::new(&model, &source, rand::rng()),
        );
        assert_eq!(output.buffer(), &[1, 2, 3, 4]);
    }
}
