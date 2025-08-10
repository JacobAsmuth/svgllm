from __future__ import annotations

import numpy as np

from svgllm.renderer import render_svg_to_rgb
from svgllm.reward import reward_svg_against_target


def main() -> None:
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256' viewBox='0 0 256 256'>"
        "<rect x='0' y='0' width='256' height='256' fill='white'/>"
        "<circle cx='128' cy='128' r='64' fill='red' stroke='black' stroke-width='4'/>"
        "</svg>"
    )

    target = render_svg_to_rgb(svg, size=(256, 256))
    # Compare the SVG to itself; reward should be close to 1 minus tiny length penalty
    reward = reward_svg_against_target(
        svg_text=svg,
        target_rgb=target,
        size=(256, 256),
        token_count=len(svg),
    )
    print(f"Reward (self-compare): {reward:.4f}")

    # Now compare against a slightly different SVG
    svg2 = svg.replace("red", "blue")
    render2 = render_svg_to_rgb(svg2, size=(256, 256))
    mse = float(np.mean(((target.astype(np.float32) - render2.astype(np.float32)) / 255.0) ** 2))
    reward2 = reward_svg_against_target(
        svg_text=svg2,
        target_rgb=target,
        size=(256, 256),
        token_count=len(svg2),
    )
    print(f"MSE(target, render2): {mse:.6f}")
    print(f"Reward (color-changed): {reward2:.4f}")


if __name__ == "__main__":
    main()



