use crate::generators::GridGenetator;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, ops::Range};
use vek::Vec2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GridDirection {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

impl GridDirection {
    pub fn opposite(&self) -> Self {
        match self {
            GridDirection::North => GridDirection::South,
            GridDirection::NorthEast => GridDirection::SouthWest,
            GridDirection::East => GridDirection::West,
            GridDirection::SouthEast => GridDirection::NorthWest,
            GridDirection::South => GridDirection::North,
            GridDirection::SouthWest => GridDirection::NorthEast,
            GridDirection::West => GridDirection::East,
            GridDirection::NorthWest => GridDirection::SouthEast,
        }
    }
}

impl std::fmt::Display for GridDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            GridDirection::North => "N",
            GridDirection::NorthEast => "NE",
            GridDirection::East => "E",
            GridDirection::SouthEast => "SE",
            GridDirection::South => "S",
            GridDirection::SouthWest => "SW",
            GridDirection::West => "W",
            GridDirection::NorthWest => "NW",
        };
        write!(f, "{}", name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grid<T: Copy> {
    size: Vec2<usize>,
    buffer: Vec<T>,
}

impl<T: Copy> Grid<T> {
    pub fn new(size: impl Into<Vec2<usize>>, fill_value: T) -> Self {
        let size = size.into();
        Self {
            size,
            buffer: vec![fill_value; size.x * size.y],
        }
    }

    pub fn with_buffer(size: impl Into<Vec2<usize>>, buffer: Vec<T>) -> Option<Self> {
        let size = size.into();
        if buffer.len() == size.x * size.y {
            Some(Self { size, buffer })
        } else {
            None
        }
    }

    pub fn view(&self, range: Range<Vec2<usize>>) -> GridView<'_, T> {
        GridView::new(self, range)
    }

    pub fn view_mut(&mut self, range: Range<Vec2<usize>>) -> GridViewMut<'_, T> {
        GridViewMut::new(self, range)
    }

    pub fn generate(size: impl Into<Vec2<usize>>, generator: impl GridGenetator<T>) -> Self
    where
        T: Default,
    {
        let mut result = Self::new(size, Default::default());
        result.apply_all(generator);
        result
    }

    pub fn fork(&self, fill_value: T) -> Self {
        Self {
            size: self.size,
            buffer: vec![fill_value; self.size.x * self.size.y],
        }
    }

    pub fn fork_generate(&self, generator: impl GridGenetator<T>) -> Self {
        let mut result = self.clone();
        result.apply_all(generator);
        result
    }

    pub fn apply(
        &mut self,
        from: impl Into<Vec2<usize>>,
        to: impl Into<Vec2<usize>>,
        mut generator: impl GridGenetator<T>,
    ) {
        if self.buffer.is_empty() {
            return;
        }
        let from = from.into();
        let to = to.into();
        for y in from.y..to.y {
            for x in from.x..to.x {
                let location = Vec2::new(x, y);
                let index = self.index(location);
                self.buffer[index] =
                    generator.generate(location, self.size, self.buffer[index], self);
            }
        }
    }

    pub fn apply_all(&mut self, generator: impl GridGenetator<T>) {
        self.apply(0, self.size, generator);
    }

    pub fn map<U: Copy>(&self, mut f: impl FnMut(Vec2<usize>, Vec2<usize>, T) -> U) -> Grid<U> {
        Grid {
            size: self.size,
            buffer: self
                .buffer
                .iter()
                .enumerate()
                .map(|(index, value)| f(self.location(index), self.size, *value))
                .collect(),
        }
    }

    pub fn into_inner(self) -> (Vec2<usize>, Vec<T>) {
        (self.size, self.buffer)
    }

    pub fn size(&self) -> Vec2<usize> {
        self.size
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn buffer(&self) -> &[T] {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut [T] {
        &mut self.buffer
    }

    pub fn iter(&self) -> impl Iterator<Item = (Vec2<usize>, usize, T)> + '_ {
        self.buffer
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| (self.location(index), index, value))
    }

    pub fn index(&self, location: impl Into<Vec2<usize>>) -> usize {
        let location = location.into();
        (location.y % self.size.y) * self.size.x + (location.x % self.size.x)
    }

    pub fn location(&self, index: usize) -> Vec2<usize> {
        Vec2 {
            x: index % self.size.x,
            y: (index / self.size.y) % self.size.y,
        }
    }

    pub fn location_offset(
        &self,
        location: impl Into<Vec2<usize>>,
        direction: GridDirection,
        distance: usize,
    ) -> Option<Vec2<usize>> {
        if distance == 0 {
            return None;
        }
        let mut location = location.into();
        match direction {
            GridDirection::North => {
                if let Some(y) = location.y.checked_sub(distance) {
                    location.y = y;
                } else {
                    return None;
                }
            }
            GridDirection::NorthEast => {
                if location.x + distance < self.size.x {
                    location.x += distance;
                } else {
                    return None;
                }
                if let Some(y) = location.y.checked_sub(distance) {
                    location.y = y;
                } else {
                    return None;
                }
            }
            GridDirection::East => {
                if location.x + distance < self.size.x {
                    location.x += distance;
                } else {
                    return None;
                }
            }
            GridDirection::SouthEast => {
                if location.x + distance < self.size.x {
                    location.x += distance;
                } else {
                    return None;
                }
                if location.y + distance < self.size.y {
                    location.y += distance;
                } else {
                    return None;
                }
            }
            GridDirection::South => {
                if location.y + distance < self.size.y {
                    location.y += distance;
                } else {
                    return None;
                }
            }
            GridDirection::SouthWest => {
                if let Some(x) = location.x.checked_sub(distance) {
                    location.x = x;
                } else {
                    return None;
                }
                if location.y + distance < self.size.y {
                    location.y += distance;
                } else {
                    return None;
                }
            }
            GridDirection::West => {
                if let Some(x) = location.x.checked_sub(distance) {
                    location.x = x;
                } else {
                    return None;
                }
            }
            GridDirection::NorthWest => {
                if let Some(x) = location.x.checked_sub(distance) {
                    location.x = x;
                } else {
                    return None;
                }
                if let Some(y) = location.y.checked_sub(distance) {
                    location.y = y;
                } else {
                    return None;
                }
            }
        }
        Some(location)
    }

    pub fn neighbors(
        &self,
        location: impl Into<Vec2<usize>>,
        range: Range<usize>,
    ) -> impl Iterator<Item = (GridDirection, Vec2<usize>, T)> + '_ {
        let location = location.into();
        range.flat_map(move |distance| {
            [
                GridDirection::North,
                GridDirection::NorthEast,
                GridDirection::East,
                GridDirection::SouthEast,
                GridDirection::South,
                GridDirection::SouthWest,
                GridDirection::West,
                GridDirection::NorthWest,
            ]
            .into_iter()
            .filter_map(move |direction| {
                let location = self.location_offset(location, direction, distance)?;
                Some((direction, location, self.get(location)?))
            })
        })
    }

    pub fn get(&self, location: impl Into<Vec2<usize>>) -> Option<T> {
        let index = self.index(location);
        self.buffer.get(index).copied()
    }

    pub fn set(&mut self, location: impl Into<Vec2<usize>>, value: T) {
        let index = self.index(location);
        if let Some(item) = self.buffer.get_mut(index) {
            *item = value;
        }
    }

    pub fn mirrored(&self, vertical: bool) -> Option<Self> {
        let mut buffer = Vec::with_capacity(self.buffer.len());
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                let src_x = if vertical { self.size.x - 1 - x } else { x };
                let src_y = if vertical { y } else { self.size.y - 1 - y };
                let index = self.index(Vec2::new(src_x, src_y));
                buffer.push(self.buffer[index]);
            }
        }
        Self::with_buffer(self.size, buffer)
    }

    pub fn rotated(&self, clockwise: bool) -> Option<Self> {
        let mut buffer = Vec::with_capacity(self.buffer.len());
        for y in 0..self.size.x {
            for x in 0..self.size.y {
                let src_x = if clockwise { y } else { self.size.x - 1 - y };
                let src_y = if clockwise { self.size.y - 1 - x } else { x };
                let index = self.index(Vec2::new(src_x, src_y));
                buffer.push(self.buffer[index]);
            }
        }
        let new_size = Vec2::new(self.size.y, self.size.x);
        Self::with_buffer(new_size, buffer)
    }
}

impl<T: Copy + Display> std::fmt::Display for Grid<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                let index = self.index(Vec2::new(x, y));
                write!(f, "{} ", self.buffer[index])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub struct GridView<'a, T: Copy> {
    grid: &'a Grid<T>,
    range: Range<Vec2<usize>>,
}

impl<'a, T: Copy> GridView<'a, T> {
    pub fn new(grid: &'a Grid<T>, range: Range<Vec2<usize>>) -> Self {
        Self { grid, range }
    }

    pub fn grid(&self) -> &'a Grid<T> {
        self.grid
    }

    pub fn range(&self) -> Range<Vec2<usize>> {
        self.range.clone()
    }

    pub fn size(&self) -> Vec2<usize> {
        Vec2::new(
            self.range.end.x - self.range.start.x,
            self.range.end.y - self.range.start.y,
        )
    }

    pub fn len(&self) -> usize {
        self.size().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = (Vec2<usize>, usize, T)> + '_ {
        self.grid
            .iter()
            .filter_map(move |(location, index, value)| {
                if location.x >= self.range.start.x
                    && location.x < self.range.end.x
                    && location.y >= self.range.start.y
                    && location.y < self.range.end.y
                {
                    let local_location = Vec2::new(
                        location.x - self.range.start.x,
                        location.y - self.range.start.y,
                    );
                    Some((local_location, index, value))
                } else {
                    None
                }
            })
    }

    pub fn grid_to_local(&self, location: impl Into<Vec2<usize>>) -> Option<Vec2<usize>> {
        let location = location.into();
        if location.x >= self.range.start.x
            && location.x < self.range.end.x
            && location.y >= self.range.start.y
            && location.y < self.range.end.y
        {
            Some(Vec2::new(
                location.x - self.range.start.x,
                location.y - self.range.start.y,
            ))
        } else {
            None
        }
    }

    pub fn local_to_grid(&self, location: impl Into<Vec2<usize>>) -> Option<Vec2<usize>> {
        let location = location.into();
        if location.x < self.size().x && location.y < self.size().y {
            Some(Vec2::new(
                location.x + self.range.start.x,
                location.y + self.range.start.y,
            ))
        } else {
            None
        }
    }
}

impl<'a, T: Copy + Display> std::fmt::Display for GridView<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.size().y {
            for x in 0..self.size().x {
                let location = self.local_to_grid(Vec2::new(x, y)).unwrap();
                let index = self.grid.index(location);
                write!(f, "{} ", self.grid.get(index).unwrap())?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub struct GridViewMut<'a, T: Copy> {
    grid: &'a mut Grid<T>,
    range: Range<Vec2<usize>>,
}

impl<'a, T: Copy> GridViewMut<'a, T> {
    pub fn new(grid: &'a mut Grid<T>, range: Range<Vec2<usize>>) -> Self {
        Self { grid, range }
    }

    pub fn grid(&mut self) -> &mut Grid<T> {
        self.grid
    }

    pub fn range(&self) -> Range<Vec2<usize>> {
        self.range.clone()
    }

    pub fn size(&self) -> Vec2<usize> {
        Vec2::new(
            self.range.end.x - self.range.start.x,
            self.range.end.y - self.range.start.y,
        )
    }

    pub fn len(&self) -> usize {
        self.size().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = (Vec2<usize>, usize, T)> + '_ {
        self.grid
            .iter()
            .filter_map(move |(location, index, value)| {
                if location.x >= self.range.start.x
                    && location.x < self.range.end.x
                    && location.y >= self.range.start.y
                    && location.y < self.range.end.y
                {
                    let local_location = Vec2::new(
                        location.x - self.range.start.x,
                        location.y - self.range.start.y,
                    );
                    Some((local_location, index, value))
                } else {
                    None
                }
            })
    }

    pub fn grid_to_local(&self, location: impl Into<Vec2<usize>>) -> Option<Vec2<usize>> {
        let location = location.into();
        if location.x >= self.range.start.x
            && location.x < self.range.end.x
            && location.y >= self.range.start.y
            && location.y < self.range.end.y
        {
            Some(Vec2::new(
                location.x - self.range.start.x,
                location.y - self.range.start.y,
            ))
        } else {
            None
        }
    }

    pub fn local_to_grid(&self, location: impl Into<Vec2<usize>>) -> Option<Vec2<usize>> {
        let location = location.into();
        if location.x < self.size().x && location.y < self.size().y {
            Some(Vec2::new(
                location.x + self.range.start.x,
                location.y + self.range.start.y,
            ))
        } else {
            None
        }
    }
}

impl<'a, T: Copy + Display> std::fmt::Display for GridViewMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.size().y {
            for x in 0..self.size().x {
                let location = self.local_to_grid(Vec2::new(x, y)).unwrap();
                let index = self.grid.index(location);
                write!(f, "{} ", self.grid.get(index).unwrap())?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::zero_prefixed_literal)]
    use super::*;

    #[test]
    fn test_mirrored() {
        let grid = Grid::with_buffer(
            Vec2::new(3, 3),
            vec![
                00, 01, 02, //
                10, 11, 12, //
                20, 21, 22, //
            ],
        )
        .unwrap();

        let mirrored = grid.mirrored(false).unwrap();
        assert_eq!(
            mirrored.buffer(),
            &[
                20, 21, 22, //
                10, 11, 12, //
                00, 01, 02, //
            ]
        );

        let mirrored = mirrored.mirrored(false).unwrap();
        assert_eq!(
            mirrored.buffer(),
            &[
                00, 01, 02, //
                10, 11, 12, //
                20, 21, 22, //
            ]
        );

        let mirrored = grid.mirrored(true).unwrap();
        assert_eq!(
            mirrored.buffer(),
            &[
                02, 01, 00, //
                12, 11, 10, //
                22, 21, 20, //
            ]
        );

        let mirrored = mirrored.mirrored(true).unwrap();
        assert_eq!(
            mirrored.buffer(),
            &[
                00, 01, 02, //
                10, 11, 12, //
                20, 21, 22, //
            ]
        );
    }

    #[test]
    fn test_rotated() {
        let grid = Grid::with_buffer(
            Vec2::new(3, 3),
            vec![
                00, 01, 02, //
                10, 11, 12, //
                20, 21, 22, //
            ],
        )
        .unwrap();

        let rotated = grid.rotated(true).unwrap();
        assert_eq!(
            rotated.buffer(),
            &[
                20, 10, 00, //
                21, 11, 01, //
                22, 12, 02, //
            ]
        );

        let rotated = rotated.rotated(true).unwrap();
        assert_eq!(
            rotated.buffer(),
            &[
                22, 21, 20, //
                12, 11, 10, //
                02, 01, 00, //
            ]
        );

        let rotated = rotated.rotated(true).unwrap();
        assert_eq!(
            rotated.buffer(),
            &[
                02, 12, 22, //
                01, 11, 21, //
                00, 10, 20, //
            ]
        );

        let rotated = rotated.rotated(true).unwrap();
        assert_eq!(
            rotated.buffer(),
            &[
                00, 01, 02, //
                10, 11, 12, //
                20, 21, 22, //
            ]
        );

        let rotated = grid.rotated(false).unwrap();
        assert_eq!(
            rotated.buffer(),
            &[
                02, 12, 22, //
                01, 11, 21, //
                00, 10, 20, //
            ]
        );

        let rotated = rotated.rotated(false).unwrap();
        assert_eq!(
            rotated.buffer(),
            &[
                22, 21, 20, //
                12, 11, 10, //
                02, 01, 00, //
            ]
        );

        let rotated = rotated.rotated(false).unwrap();
        assert_eq!(
            rotated.buffer(),
            &[
                20, 10, 00, //
                21, 11, 01, //
                22, 12, 02, //
            ]
        );

        let rotated = rotated.rotated(false).unwrap();
        assert_eq!(
            rotated.buffer(),
            &[
                00, 01, 02, //
                10, 11, 12, //
                20, 21, 22, //
            ]
        );
    }

    #[test]
    fn test_view() {
        let a = Grid::with_buffer(
            (3, 3),
            vec![
                00, 10, 20, //
                01, 11, 21, //
                02, 12, 22, //
            ],
        )
        .unwrap();

        let b = Grid::with_buffer(
            (3, 3),
            vec![
                10, 20, 30, //
                11, 21, 31, //
                12, 22, 32, //
            ],
        )
        .unwrap();

        let a_view = a.view(Vec2::new(1, 1)..Vec2::new(3, 3));
        let b_view = b.view(Vec2::new(0, 1)..Vec2::new(2, 3));

        for ((_, _, a_value), (_, _, b_value)) in a_view.iter().zip(b_view.iter()) {
            assert_eq!(a_value, b_value);
        }
    }
}
