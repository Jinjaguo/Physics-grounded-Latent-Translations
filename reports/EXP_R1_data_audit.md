# EXP_R1 data audit

Created: 2026-08-16T03:46:06-04:00

The audit found **407** physically contiguous, reset-free 128-frame windows from the official Wave27 human-play archive. Every boundary is exactly at the midpoint (`start + 64`), so each case contains four pre-boundary and four post-boundary H16 chunks.

| split | sessions | windows | goals |
|---|---:|---:|---|
| train | 36 | 234 | lift_blue_block_slider:50, lift_red_block_table:56, place_in_slider:47, push_pink_block_right:23, turn_off_lightbulb:29, turn_on_lightbulb:29 |
| development | 8 | 87 | lift_blue_block_slider:13, lift_red_block_table:15, place_in_slider:17, push_pink_block_right:18, turn_off_lightbulb:12, turn_on_lightbulb:12 |
| heldout | 8 | 86 | lift_blue_block_slider:15, lift_red_block_table:12, place_in_slider:15, push_pink_block_right:13, turn_off_lightbulb:15, turn_on_lightbulb:16 |

The planner receives only the start latent, target task text/region, and frozen model interfaces. The four post-boundary chunks remain hidden until metric evaluation. Target regions are constructed from train post-boundary chunks only. Source-session separation is preserved.
