#![cfg(feature = "diagnostics")]

use crate::{
    grid::GridDirection,
    wfc::{Pattern, PatternId, WfcModel},
};
use html_builder::{Buffer, Html5, Node};
use std::{error::Error, fmt::Write};

impl<T: Copy> Pattern<T> {
    pub fn diagnostics(&self, container: &mut Node) -> Result<(), Box<dyn Error>>
    where
        T: std::fmt::Display,
    {
        let mut table = container
            .table()
            .attr(r#"class="pattern""#)
            .attr(&format!(r#"id="pattern-{}""#, self.id.0));

        {
            table
                .tr()
                .td()
                .attr(r#"class="pattern-info""#)
                .attr(&format!(r#"colspan="{}""#, self.grid.size().x))
                .raw()
                .write_str(&format!(
                    "ID: {}</br>Frequency: {}</br>Weight: {:.3}",
                    self.id.0, self.frequency, self.weight
                ))?;
        }

        for y in 0..self.grid.size().y {
            let mut row = table.tr();
            for x in 0..self.grid.size().x {
                let cell = self.grid.get((x, y)).unwrap();
                row.td().write_str(&format!("{}", cell))?;
            }
        }

        Ok(())
    }
}

impl<const N: usize, T: Copy + Eq + Ord> WfcModel<N, T> {
    pub fn diagnostics(&self) -> Result<String, Box<dyn Error>>
    where
        T: std::fmt::Display,
    {
        let mut buffer = Buffer::new();
        buffer.doctype();

        let mut html = buffer.html().attr(r#"lang="en""#);

        {
            let mut head = html.head();
            head.meta().attr(r#"charset="utf-8""#);
            head.title().write_str("WFC Diagnostics")?;

            let css = r#"
            body {
                font-family: sans-serif;
                margin: 0;
                padding: 0;
            }

            div.patterns {
                display: flex;
                flex-wrap: wrap;
                gap: 30px;
                width: 100%;
                box-sizing: border-box;
                margin: 0px;
                padding: 10px;
                position: relative;
            }
            
            table.pattern {
                border: 1px solid #666;
                border-radius: 5px;
                margin: 0px;
                padding: 5px;
                background: #fafafa;
                font-size: 15px;
                text-align: center;
                transition: opacity 0.2s;
            }

            table.pattern.highlighted {
                box-shadow: 0 0 10px 3px rgba(0, 0, 0, 0.3);
                transition: box-shadow 0.2s, opacity 0.2s;
            }

            .pattern-info {
                font-size: 12px;
                font-weight: bold;
                text-align: left;
            }

            table.pattern td {
                border: 1px solid #666;
                border-radius: 5px;
                padding: 2px;
                background: #fdfdfd;
            }
            
            svg.connections {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                overflow: visible;
                opacity: 0.35;
            }

            svg.connections line {
                transition: opacity 0.2s;
            }
            "#;
            head.style().write_str(css)?;
        }

        {
            let mut body = html.body();

            // Outer table as grid container
            let mut patterns = body.div().attr(r#"class="patterns""#);
            {
                for pattern in &self.patterns {
                    pattern.diagnostics(&mut patterns)?;
                }
            }

            // SVG overlay
            {
                let mut svg_container = body
                    .svg()
                    .attr(r#"class="connections""#)
                    .attr(r#"xmlns="http://www.w3.org/2000/svg""#)
                    .raw();

                // Inject entire SVG content as a raw string
                let svg_content = r##"
                <defs>
                    <marker
                        id="arrow-end"
                        markerWidth="4"
                        markerHeight="4"
                        refX="4"
                        refY="2"
                        orient="auto"
                        markerUnits="strokeWidth"
                    >
                        <path d="M0,0 L0,4 L4,2 z" fill="#000" />
                    </marker>
                    <marker
                        id="arrow-start"
                        markerWidth="4"
                        markerHeight="4"
                        refX="-6"
                        refY="2"
                        orient="auto"
                        markerUnits="strokeWidth"
                    >
                        <path d="M0,0 L0,4 L4,2 z" fill="#000" />
                    </marker>
                </defs>
                "##;

                svg_container.write_str(svg_content)?;
            }

            // Script block
            let mut script_body = String::from(
                r#"
                const edges = [];

                function connect(fromId, toId, fromSide, toSide, color="black") {
                    edges.push({fromId, toId, fromSide, toSide, color});
                }

                function getPortCoordinates(el, side, index = 0, total = 1, portion = 0.8) {
                    const rect = el.getBoundingClientRect();
                    let x, y;

                    if (side === "N") { // top
                        x = rect.left + rect.width / 2;
                        y = rect.top;
                    } else if (side === "S") { // bottom
                        x = rect.left + rect.width / 2;
                        y = rect.bottom;
                    } else if (side === "E") { // right
                        x = rect.right;
                        y = rect.top + rect.height / 2;
                    } else if (side === "W") { // left
                        x = rect.left;
                        y = rect.top + rect.height / 2;
                    } else {
                        x = rect.left + rect.width / 2;
                        y = rect.top + rect.height / 2;
                    }

                    // apply spreading if total > 1
                    if (total > 1) {
                        if (side === "N" || side === "S") {
                            const usableWidth = rect.width * portion;
                            const startX = rect.left + (rect.width - usableWidth) / 2;
                            x = startX + ((index + 0.5) / total) * usableWidth;
                        }
                        if (side === "E" || side === "W") {
                            const usableHeight = rect.height * portion;
                            const startY = rect.top + (rect.height - usableHeight) / 2;
                            y = startY + ((index + 0.5) / total) * usableHeight;
                        }
                    }

                    return { x, y };
                }

                function updateArrows() {
                    const svg = document.querySelector(".connections");
                    svg.innerHTML = svg.querySelector("defs").outerHTML; // preserve arrowhead

                    const pt = svg.createSVGPoint();
                    function toSvgCoords(x, y) {
                        pt.x = x;
                        pt.y = y;
                        return pt.matrixTransform(svg.getScreenCTM().inverse());
                    }

                    const counts = {};
                    edges.forEach(e => {
                        const fromKey = `${e.fromId}_${e.fromSide}`;
                        counts[fromKey] = (counts[fromKey] || 0) + 1;
                        const toKey = `${e.toId}_${e.toSide}`;
                        counts[toKey] = (counts[toKey] || 0) + 1;
                    });

                    const sideUsage = {};

                    edges.forEach(e => {
                        const fromKey = `${e.fromId}_${e.fromSide}`;
                        const fromIndex = sideUsage[fromKey] || 0;
                        const fromTotal = counts[fromKey];

                        const toKey = `${e.toId}_${e.toSide}`;
                        const toIndex = sideUsage[toKey] || 0;
                        const toTotal = counts[toKey];

                        const startCoord = getPortCoordinates(
                            document.getElementById(e.fromId), e.fromSide, fromIndex, fromTotal
                        );
                        const endCoord = getPortCoordinates(
                            document.getElementById(e.toId), e.toSide, toIndex, toTotal
                        );

                        sideUsage[fromKey] = fromIndex + 1;
                        sideUsage[toKey] = toIndex + 1;

                        const start = toSvgCoords(startCoord.x, startCoord.y);
                        const end = toSvgCoords(endCoord.x, endCoord.y);

                        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                        line.setAttribute("x1", start.x);
                        line.setAttribute("y1", start.y);
                        line.setAttribute("x2", end.x);
                        line.setAttribute("y2", end.y);
                        line.setAttribute("stroke", e.color);
                        line.setAttribute("stroke-width", "1.5");
                        line.setAttribute("marker-end", "url(#arrow-end)");
                        line.setAttribute("marker-start", "url(#arrow-start)");

                        // tag for hover filtering
                        line.dataset.from = e.fromId;
                        line.dataset.to = e.toId;
                        line.dataset.fromSide = e.fromSide;
                        line.dataset.toSide = e.toSide;

                        svg.appendChild(line);
                    });

                    document.querySelectorAll(".pattern").forEach(pattern => {
                        pattern.addEventListener("mousemove", event => {
                            const id = pattern.id;
                            const rect = pattern.getBoundingClientRect();
                            const offsetX = event.clientX - rect.left;
                            const offsetY = event.clientY - rect.top;
                            const w = rect.width;
                            const h = rect.height;

                            const margin = 10;
                            let activeSide = null;

                            if (offsetY < margin) activeSide = 'N';
                            else if (offsetY > h - margin) activeSide = 'S';
                            else if (offsetX < margin) activeSide = 'W';
                            else if (offsetX > w - margin) activeSide = 'E';
                            // if cursor is deeper inside the pattern, activeSide stays null → show all sides

                            document.querySelectorAll(".connections line").forEach(line => {
                                if (!activeSide) {
                                    // full pattern hover, show all connected lines
                                    line.style.opacity = (line.dataset.from === id || line.dataset.to === id) ? "1" : "0";
                                } else {
                                    // side-specific hover
                                    const matches = (line.dataset.from === id && line.dataset.fromSide === activeSide)
                                        || (line.dataset.to === id && line.dataset.toSide === activeSide);
                                    line.style.opacity = matches ? "1" : "0";
                                }
                            });

                            // also fade unrelated patterns
                            document.querySelectorAll(".pattern").forEach(p => {
                                p.style.opacity = (p.id === id) ? "1" : "0.5";
                            });
                        });

                        pattern.addEventListener("mouseleave", () => {
                            // restore full visibility
                            document.querySelectorAll(".connections line").forEach(line => {
                                line.style.opacity = "1";
                            });
                            document.querySelectorAll(".pattern").forEach(p => {
                                p.style.opacity = "1";
                            });
                        });
                    });
                }

                function filterPatterns(filterIds) {
                    const allPatterns = document.querySelectorAll(".pattern");
                    const allLines = document.querySelectorAll(".connections line");

                    if (!filterIds || filterIds.length === 0) {
                        // show everything
                        allPatterns.forEach(p => p.style.display = "block");
                        allLines.forEach(l => l.style.display = "block");
                        return;
                    }

                    // first, show only patterns that match the filter or are connected
                    const connectedPatterns = new Set(filterIds);

                    // add any patterns directly connected to filtered patterns
                    allLines.forEach(line => {
                        if (filterIds.includes(line.dataset.from)) connectedPatterns.add(line.dataset.to);
                        if (filterIds.includes(line.dataset.to)) connectedPatterns.add(line.dataset.from);
                    });

                    // show/hide patterns
                    allPatterns.forEach(p => {
                        p.style.display = connectedPatterns.has(p.id) ? "block" : "none";
                    });

                    // show/hide lines (only lines connecting visible patterns)
                    allLines.forEach(line => {
                        if (connectedPatterns.has(line.dataset.from) && connectedPatterns.has(line.dataset.to)) {
                            line.style.display = "block";
                        } else {
                            line.style.display = "none";
                        }
                    });
                }

                window.addEventListener("resize", updateArrows);
                window.addEventListener("scroll", updateArrows);
                "#,
            );

            for (&(PatternId(from_index), direction), to_bits) in &self.compatibility {
                for to_index in to_bits.iter_ones() {
                    let from_side = match direction {
                        GridDirection::North => "N",
                        GridDirection::East => "E",
                        GridDirection::South => "S",
                        GridDirection::West => "W",
                        _ => "C",
                    };
                    let to_side = match direction.opposite() {
                        GridDirection::North => "N",
                        GridDirection::East => "E",
                        GridDirection::South => "S",
                        GridDirection::West => "W",
                        _ => "C",
                    };
                    let color = match direction {
                        GridDirection::North => "rgb(255, 0, 0)",
                        GridDirection::East => "rgb(0, 180, 0)",
                        GridDirection::South => "rgb(60, 60, 255)",
                        GridDirection::West => "rgb(240, 0, 240)",
                        _ => "black",
                    };
                    writeln!(
                        script_body,
                        r#"connect("pattern-{}", "pattern-{}", "{}", "{}", "{}");"#,
                        from_index, to_index, from_side, to_side, color
                    )?;
                }
            }
            writeln!(script_body, "updateArrows();")?;

            body.script().raw().write_str(&script_body)?;
        }

        Ok(buffer.finish())
    }
}
