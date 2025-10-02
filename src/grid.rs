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
        GridView::new(self.size().x, &self.buffer, range)
    }

    pub fn view_mut(&mut self, range: Range<Vec2<usize>>) -> GridViewMut<'_, T> {
        GridViewMut::new(self.size().x, &mut self.buffer, range)
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

    pub fn apply_at(
        &mut self,
        locations: impl IntoIterator<Item = Vec2<usize>>,
        mut generator: impl GridGenetator<T>,
    ) {
        for location in locations {
            let index = self.index(location);
            self.buffer[index] = generator.generate(location, self.size, self.buffer[index], self);
        }
    }

    pub fn apply(
        &mut self,
        from: impl Into<Vec2<usize>>,
        to: impl Into<Vec2<usize>>,
        generator: impl GridGenetator<T>,
    ) {
        if self.buffer.is_empty() {
            return;
        }
        let from = from.into();
        let to = to.into();
        self.apply_at(
            (from.y..to.y).flat_map(|y| (from.x..to.x).map(move |x| Vec2::new(x, y))),
            generator,
        );
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
        location.y * self.size.x + location.x
    }

    pub fn safe_index(&self, location: impl Into<Vec2<usize>>) -> Option<usize> {
        let location = location.into();
        if location.x < self.size.x && location.y < self.size.y {
            Some(location.y * self.size.x + location.x)
        } else {
            None
        }
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
                let (src_x, src_y) = if vertical {
                    (x, self.size.y - 1 - y)
                } else {
                    (self.size.x - 1 - x, y)
                };
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
                let (src_x, src_y) = if clockwise {
                    (y, self.size.x - 1 - x)
                } else {
                    (self.size.y - 1 - y, x)
                };
                let index = self.index(Vec2::new(src_x, src_y));
                buffer.push(self.buffer[index]);
            }
        }
        let new_size = Vec2::new(self.size.y, self.size.x);
        Self::with_buffer(new_size, buffer)
    }

    pub fn shifted(&self, direction: GridDirection) -> Option<Self> {
        let mut buffer = Vec::with_capacity(self.buffer.len());
        for y in 0..self.size.x {
            for x in 0..self.size.y {
                let src = self.location_offset(Vec2::new(x, y), direction.opposite(), 1);
                let value = if let Some(src) = src {
                    self.buffer[src.y * self.size.x + src.x]
                } else {
                    self.buffer[y * self.size.x + x]
                };
                buffer.push(value);
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FixedGrid<const W: usize, const H: usize, T: Copy> {
    buffer: [[T; W]; H],
}

impl<const W: usize, const H: usize, T: Copy> FixedGrid<W, H, T> {
    pub fn new(fill_value: T) -> Self {
        Self {
            buffer: [[fill_value; W]; H],
        }
    }

    pub fn with_buffer(buffer: [[T; W]; H]) -> Self {
        Self { buffer }
    }

    pub fn view(&self, range: Range<Vec2<usize>>) -> GridView<'_, T> {
        GridView::new(self.size().x, self.as_slice(), range)
    }

    pub fn view_mut(&mut self, range: Range<Vec2<usize>>) -> GridViewMut<'_, T> {
        GridViewMut::new(self.size().x, self.as_mut_slice(), range)
    }

    pub fn map<U: Copy>(&self, mut f: impl FnMut(Vec2<usize>, T) -> U) -> FixedGrid<W, H, U> {
        FixedGrid {
            buffer: std::array::from_fn(|y| {
                std::array::from_fn(|x| f(Vec2::new(x, y), self.buffer[y][x]))
            }),
        }
    }

    pub fn into_inner(self) -> [[T; W]; H] {
        self.buffer
    }

    pub fn size(&self) -> Vec2<usize> {
        Vec2::new(W, H)
    }

    pub fn len(&self) -> usize {
        W * H
    }

    pub fn is_empty(&self) -> bool {
        W == 0 || H == 0
    }

    pub fn buffer(&self) -> &[[T; W]; H] {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut [[T; W]; H] {
        &mut self.buffer
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr() as *const T, W * H) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr() as *mut T, W * H) }
    }

    pub fn iter(&self) -> impl Iterator<Item = (Vec2<usize>, usize, T)> + '_ {
        (0..H)
            .flat_map(move |y| (0..W).map(move |x| (Vec2::new(x, y), y * W + x, self.buffer[y][x])))
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
                if location.x + distance < W {
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
                if location.x + distance < W {
                    location.x += distance;
                } else {
                    return None;
                }
            }
            GridDirection::SouthEast => {
                if location.x + distance < W {
                    location.x += distance;
                } else {
                    return None;
                }
                if location.y + distance < H {
                    location.y += distance;
                } else {
                    return None;
                }
            }
            GridDirection::South => {
                if location.y + distance < H {
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
                if location.y + distance < H {
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
                Some((direction, location, self.buffer[location.y][location.x]))
            })
        })
    }

    pub fn get(&self, location: impl Into<Vec2<usize>>) -> Option<T> {
        let location = location.into();
        if location.x < W && location.y < H {
            Some(self.buffer[location.y][location.x])
        } else {
            None
        }
    }

    pub fn set(&mut self, location: impl Into<Vec2<usize>>, value: T) {
        let location = location.into();
        if location.x < W && location.y < H {
            self.buffer[location.y][location.x] = value;
        }
    }

    pub fn mirrored(&self, vertical: bool) -> Self {
        let mut buffer = [[self.buffer[0][0]; W]; H];
        for (y, row) in buffer.iter_mut().enumerate() {
            for (x, cell) in row.iter_mut().enumerate() {
                let (src_x, src_y) = if vertical {
                    (x, H - 1 - y)
                } else {
                    (W - 1 - x, y)
                };
                *cell = self.buffer[src_y][src_x];
            }
        }
        Self { buffer }
    }

    pub fn rotated(&self, clockwise: bool) -> FixedGrid<H, W, T> {
        let mut buffer = [[self.buffer[0][0]; H]; W];
        for (y, row) in buffer.iter_mut().enumerate() {
            for (x, cell) in row.iter_mut().enumerate() {
                let (src_x, src_y) = if clockwise {
                    (y, H - 1 - x)
                } else {
                    (W - 1 - y, x)
                };
                *cell = self.buffer[src_y][src_x];
            }
        }
        FixedGrid { buffer }
    }

    pub fn shifted(&self, direction: GridDirection) -> Self {
        let buffer = std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                let src = self.location_offset(Vec2::new(col, row), direction.opposite(), 1);
                if let Some(src) = src {
                    self.buffer[src.y][src.x]
                } else {
                    self.buffer[row][col]
                }
            })
        });
        Self { buffer }
    }
}

impl<const W: usize, const H: usize, T: Copy + Display> std::fmt::Display for FixedGrid<W, H, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..H {
            for x in 0..W {
                write!(f, "{} ", self.buffer[y][x])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub struct GridView<'a, T: Copy> {
    stride: usize,
    cells: &'a [T],
    range: Range<Vec2<usize>>,
}

impl<'a, T: Copy> GridView<'a, T> {
    pub fn new(stride: usize, cells: &'a [T], range: Range<Vec2<usize>>) -> Self {
        Self {
            stride,
            cells,
            range,
        }
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn cells(&self) -> &[T] {
        self.cells
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

    pub fn iter(&self) -> impl Iterator<Item = (Vec2<usize>, usize, &T)> + '_ {
        let Vec2 { x, y } = self.size();
        (0..y).flat_map(move |local_y| {
            (0..x).filter_map(move |local_x| {
                let local_location = Vec2::new(local_x, local_y);
                let grid_location = self.local_to_grid(local_location)?;
                let index = grid_location.y * self.stride + grid_location.x;
                Some((local_location, index, &self.cells[index]))
            })
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

    pub fn get(&self, location: impl Into<Vec2<usize>>) -> Option<&T> {
        let location = location.into();
        let grid_location = self.local_to_grid(location)?;
        let index = grid_location.y * self.stride + grid_location.x;
        self.cells.get(index)
    }
}

impl<'a, T: Copy + Display> std::fmt::Display for GridView<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.size().y {
            for x in 0..self.size().x {
                let location = self.local_to_grid(Vec2::new(x, y)).unwrap();
                let index = location.y * self.stride + location.x;
                write!(f, "{} ", self.cells[index])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub struct GridViewMut<'a, T: Copy> {
    stride: usize,
    cells: &'a mut [T],
    range: Range<Vec2<usize>>,
}

impl<'a, T: Copy> GridViewMut<'a, T> {
    pub fn new(stride: usize, cells: &'a mut [T], range: Range<Vec2<usize>>) -> Self {
        Self {
            stride,
            cells,
            range,
        }
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn cells(&self) -> &[T] {
        self.cells
    }

    pub fn cells_mut(&mut self) -> &mut [T] {
        self.cells
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

    pub fn iter(&self) -> impl Iterator<Item = (Vec2<usize>, usize, &T)> + '_ {
        let Vec2 { x, y } = self.size();
        (0..y).flat_map(move |local_y| {
            (0..x).filter_map(move |local_x| {
                let local_location = Vec2::new(local_x, local_y);
                let grid_location = self.local_to_grid(local_location)?;
                let index = grid_location.y * self.stride + grid_location.x;
                Some((local_location, index, &self.cells[index]))
            })
        })
    }

    pub fn iter_mut(&'a mut self) -> impl Iterator<Item = (Vec2<usize>, usize, &'a mut T)> + 'a {
        let stride = self.stride;
        let range = self.range.clone();
        let size = self.size();
        let cells_ptr = self.cells.as_mut_ptr();
        let cells_len = self.cells.len();
        (0..size.y).flat_map(move |local_y| {
            (0..size.x).filter_map(move |local_x| {
                let local_location = Vec2::new(local_x, local_y);
                let grid_location = Vec2::new(
                    local_location.x + range.start.x,
                    local_location.y + range.start.y,
                );
                if grid_location.x >= stride || grid_location.y * stride >= cells_len {
                    return None;
                }
                let index = grid_location.y * stride + grid_location.x;
                // SAFETY: We guarantee unique access by construction of the iterator.
                let cell = unsafe { &mut *cells_ptr.add(index) };
                Some((local_location, index, cell))
            })
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

    pub fn get(&self, location: impl Into<Vec2<usize>>) -> Option<&T> {
        let location = location.into();
        let grid_location = self.local_to_grid(location)?;
        let index = grid_location.y * self.stride + grid_location.x;
        self.cells.get(index)
    }
}

impl<'a, T: Copy + Display> std::fmt::Display for GridViewMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.size().y {
            for x in 0..self.size().x {
                let location = self.local_to_grid(Vec2::new(x, y)).unwrap();
                let index = location.y * self.stride + location.x;
                write!(f, "{} ", self.cells[index])?;
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
    fn test_mirrored_grid() {
        let grid = Grid::with_buffer(
            Vec2::new(3, 3),
            vec![
                00, 01, 02, //
                10, 11, 12, //
                20, 21, 22, //
            ],
        )
        .unwrap();

        let mirrored = grid.mirrored(true).unwrap();
        assert_eq!(
            mirrored.buffer(),
            &[
                20, 21, 22, //
                10, 11, 12, //
                00, 01, 02, //
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

        let mirrored = grid.mirrored(false).unwrap();
        assert_eq!(
            mirrored.buffer(),
            &[
                02, 01, 00, //
                12, 11, 10, //
                22, 21, 20, //
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
    }

    #[test]
    fn test_mirrored_fixed_grid() {
        let grid = FixedGrid::<3, 3, _>::with_buffer([
            [00, 01, 02], //
            [10, 11, 12], //
            [20, 21, 22], //
        ]);

        let mirrored = grid.mirrored(true);
        assert_eq!(
            mirrored.buffer(),
            &[
                [20, 21, 22], //
                [10, 11, 12], //
                [00, 01, 02], //
            ]
        );

        let mirrored = mirrored.mirrored(true);
        assert_eq!(
            mirrored.buffer(),
            &[
                [00, 01, 02], //
                [10, 11, 12], //
                [20, 21, 22], //
            ]
        );

        let mirrored = grid.mirrored(false);
        assert_eq!(
            mirrored.buffer(),
            &[
                [02, 01, 00], //
                [12, 11, 10], //
                [22, 21, 20], //
            ]
        );

        let mirrored = mirrored.mirrored(false);
        assert_eq!(
            mirrored.buffer(),
            &[
                [00, 01, 02], //
                [10, 11, 12], //
                [20, 21, 22], //
            ]
        );
    }

    #[test]
    fn test_rotated_grid() {
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
    fn test_rotated_fixed_grid() {
        let grid = FixedGrid::<3, 3, _>::with_buffer([
            [00, 01, 02], //
            [10, 11, 12], //
            [20, 21, 22], //
        ]);

        let rotated = grid.rotated(true);
        assert_eq!(
            rotated.buffer(),
            &[
                [20, 10, 00], //
                [21, 11, 01], //
                [22, 12, 02], //
            ]
        );

        let rotated = rotated.rotated(true);
        assert_eq!(
            rotated.buffer(),
            &[
                [22, 21, 20], //
                [12, 11, 10], //
                [02, 01, 00], //
            ]
        );

        let rotated = rotated.rotated(true);
        assert_eq!(
            rotated.buffer(),
            &[
                [02, 12, 22], //
                [01, 11, 21], //
                [00, 10, 20], //
            ]
        );

        let rotated = rotated.rotated(true);
        assert_eq!(
            rotated.buffer(),
            &[
                [00, 01, 02], //
                [10, 11, 12], //
                [20, 21, 22], //
            ]
        );

        let rotated = grid.rotated(false);
        assert_eq!(
            rotated.buffer(),
            &[
                [02, 12, 22], //
                [01, 11, 21], //
                [00, 10, 20], //
            ]
        );

        let rotated = rotated.rotated(false);
        assert_eq!(
            rotated.buffer(),
            &[
                [22, 21, 20], //
                [12, 11, 10], //
                [02, 01, 00], //
            ]
        );

        let rotated = rotated.rotated(false);
        assert_eq!(
            rotated.buffer(),
            &[
                [20, 10, 00], //
                [21, 11, 01], //
                [22, 12, 02], //
            ]
        );

        let rotated = rotated.rotated(false);
        assert_eq!(
            rotated.buffer(),
            &[
                [00, 01, 02], //
                [10, 11, 12], //
                [20, 21, 22], //
            ]
        );
    }

    #[test]
    fn test_shifted() {
        let a = Grid::with_buffer(
            Vec2::new(3, 3),
            vec![
                00, 01, 02, //
                10, 11, 12, //
                20, 21, 22, //
            ],
        )
        .unwrap();

        let b = a.shifted(GridDirection::North).unwrap();
        assert_eq!(
            b.buffer(),
            &[
                10, 11, 12, //
                20, 21, 22, //
                20, 21, 22, //
            ]
        );

        let b = a.shifted(GridDirection::West).unwrap();
        assert_eq!(
            b.buffer(),
            &[
                01, 02, 02, //
                11, 12, 12, //
                21, 22, 22, //
            ]
        );

        let b = a.shifted(GridDirection::South).unwrap();
        assert_eq!(
            b.buffer(),
            &[
                00, 01, 02, //
                00, 01, 02, //
                10, 11, 12, //
            ]
        );

        let b = a.shifted(GridDirection::East).unwrap();
        assert_eq!(
            b.buffer(),
            &[
                00, 00, 01, //
                10, 10, 11, //
                20, 20, 21, //
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

        let b = FixedGrid::<3, 3, _>::with_buffer([
            [10, 20, 30], //
            [11, 21, 31], //
            [12, 22, 32], //
        ]);

        let a_view = a.view(Vec2::new(1, 1)..Vec2::new(3, 3));
        let b_view = b.view(Vec2::new(0, 1)..Vec2::new(2, 3));

        for ((_, _, a_value), (_, _, b_value)) in a_view.iter().zip(b_view.iter()) {
            assert_eq!(a_value, b_value);
        }
    }
}
