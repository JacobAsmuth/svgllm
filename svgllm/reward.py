from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from skimage.metrics import structural_similarity as ssim

from .renderer import render_svg_to_rgb, is_svg_valid


def _to_float_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    return np.clip(img, 0.0, 1.0)


def simple_svg_reward(
    target_rgb: np.ndarray,
    render_rgb: np.ndarray,
    num_tokens: int,
    *,
    token_budget: int = 512,
    length_penalty: float = 0.03,
    ssim_weight: float = 0.85,
    mse_ref: float = 0.05,
    validity_bonus: float = 0.05,
) -> float:
    """Compute a simple bounded reward combining SSIM, bounded MSE, and length penalty.

    Args:
        target_rgb: (H,W,3) array in [0,255] or [0,1].
        render_rgb: (H,W,3) array in [0,255] or [0,1].
        num_tokens: Number of generated tokens (or characters as a proxy).
        token_budget: Budget for normalized length penalty denominator.
        length_penalty: Linear coefficient for length penalty.
        ssim_weight: Weight for SSIM vs MSE term.
        mse_ref: MSE value where quality is considered poor (maps to 0 contribution).
        validity_bonus: Small constant added to encourage valid outputs.
    Returns:
        Reward in [0,1].
    """
    target = _to_float_rgb(target_rgb)
    render = _to_float_rgb(render_rgb)

    ssim_score = float(ssim(target, render, channel_axis=2, data_range=1.0))
    mse = float(np.mean((target - render) ** 2))
    mse_term = 1.0 - min(1.0, mse / max(1e-8, mse_ref))

    fidelity = ssim_weight * ssim_score + (1.0 - ssim_weight) * mse_term
    brevity = length_penalty * (num_tokens / max(1, token_budget))
    reward = validity_bonus + fidelity - brevity
    return float(max(0.0, min(1.0, reward)))


def reward_svg_against_target(
    svg_text: str,
    target_rgb: np.ndarray,
    *,
    size: Tuple[int, int] = (256, 256),
    token_count: Optional[int] = None,
    token_budget: int = 512,
    length_penalty: float = 0.03,
    ssim_weight: float = 0.85,
    mse_ref: float = 0.05,
    validity_bonus: float = 0.05,
) -> float:
    """Render `svg_text`, compare to target, and compute the simple reward.

    Returns 0.0 on invalid SVG or render failure.
    """
    if not is_svg_valid(svg_text):
        return 0.0
    try:
        render = render_svg_to_rgb(svg_text, size=size)
    except Exception:
        return 0.0

    if token_count is None:
        token_count = len(svg_text)

    return simple_svg_reward(
        target_rgb=target_rgb,
        render_rgb=render,
        num_tokens=token_count,
        token_budget=token_budget,
        length_penalty=length_penalty,
        ssim_weight=ssim_weight,
        mse_ref=mse_ref,
        validity_bonus=validity_bonus,
    )


