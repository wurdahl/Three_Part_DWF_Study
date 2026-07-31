#!/usr/bin/env python3
"""Create one dependency-free SVG comparing the normalized correlators."""

import math

from correlator_io import ROOT, load_domain_wall, load_wilson


def main():
    series = [("Wilson", load_wilson()[0], "#1464a0"),
              ("Domain wall", load_domain_wall()[0], "#d1495b")]
    normalized = []
    for label, values, color in series:
        scale = max(abs(value) for value in values)
        normalized.append((label, [abs(value) / scale for value in values], color))

    width, height = 920, 560
    left, right, top, bottom = 90, 35, 60, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    max_t = max(len(values) for _, values, _ in normalized) - 1
    floor = 1.0e-14
    log_min = min(math.log10(max(value, floor))
                  for _, values, _ in normalized for value in values)

    def x(t):
        return left + plot_w * t / max_t

    def y(value):
        logged = math.log10(max(value, floor))
        return top + plot_h * (0.0 - logged) / (0.0 - log_min)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:sans-serif}.grid{stroke:#ddd}.axis{stroke:#111;stroke-width:2}.curve{fill:none;stroke-width:2}</style>",
        f"<text x='{width/2}' y='32' text-anchor='middle' font-size='23'>Dynamical fermion correlators</text>",
    ]
    for tick in range(6):
        fraction = tick / 5
        yy = top + plot_h * fraction
        label = 10 ** (log_min * fraction)
        parts += [f"<line class='grid' x1='{left}' x2='{left+plot_w}' y1='{yy}' y2='{yy}'/>",
                  f"<text x='{left-8}' y='{yy+5}' text-anchor='end' font-size='13'>{label:.1e}</text>"]
    parts += [f"<line class='axis' x1='{left}' x2='{left+plot_w}' y1='{top+plot_h}' y2='{top+plot_h}'/>",
              f"<line class='axis' x1='{left}' x2='{left}' y1='{top}' y2='{top+plot_h}'/>"]
    for index, (label, values, color) in enumerate(normalized):
        points = " ".join(f"{x(t):.2f},{y(value):.2f}" for t, value in enumerate(values))
        legend_y = 78 + index * 25
        parts += [f"<polyline class='curve' stroke='{color}' points='{points}'/>",
                  f"<line stroke='{color}' stroke-width='3' x1='{width-250}' x2='{width-210}' y1='{legend_y}' y2='{legend_y}'/>",
                  f"<text x='{width-200}' y='{legend_y+5}' font-size='15'>{label}</text>"]
    parts += [f"<text x='{left+plot_w/2}' y='{height-20}' text-anchor='middle' font-size='18'>t</text>",
              f"<text x='22' y='{top+plot_h/2}' text-anchor='middle' font-size='18' transform='rotate(-90 22 {top+plot_h/2})'>|C(t)| / max |C(t)|</text>",
              "</svg>"]
    path = ROOT / "output/correlators_comparison.svg"
    path.write_text("".join(parts))
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
