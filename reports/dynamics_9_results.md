# Twenty-first wave results: language-conditioned latent transition

Run date: 2026-08-14T17:21:32.504086-04:00

## Outcome

- C7 language-conditioned transition: **REJECTED**
- C8 language-targeted atomic transition: **REJECTED**
- Physically continuous annotation-onset transitions: **560**, sessions: **31**
- RedirectGain: 0.250126, clustered 95% CI [0.136495, 0.370798]
- Execution RedirectGain: 0.183855, clustered 95% CI [0.100917, 0.263777]
- Endpoint accuracy: 0.516260; macro=0.516260; threshold=0.60

## Required questions

1. **560** physically continuous annotation-onset transitions were found.
2. They came from **31** distinct source sessions.
3. All six requested next-goal classes met the prospective coverage gate.
4. Train/development/test counts per goal are `{'development': {'lift_blue_block_slider': 17, 'lift_red_block_table': 22, 'place_in_slider': 30, 'push_pink_block_right': 14, 'turn_off_lightbulb': 27, 'turn_on_lightbulb': 29}, 'test': {'lift_blue_block_slider': 26, 'lift_red_block_table': 27, 'place_in_slider': 37, 'push_pink_block_right': 14, 'turn_off_lightbulb': 29, 'turn_on_lightbulb': 31}, 'train': {'lift_blue_block_slider': 39, 'lift_red_block_table': 41, 'place_in_slider': 46, 'push_pink_block_right': 32, 'turn_off_lightbulb': 51, 'turn_on_lightbulb': 48}}`.
5. Yes. The representation was completely frozen; optimizer steps and EMA updates were zero.
6. Yes. The decoder was completely frozen; optimizer steps were zero and its component hash remained recorded.
7. Yes. B1 beat B0: H2 execution 0.745798 < 0.861066; H4 decoded 0.038513 < 0.044233, both clustered lower CIs positive.
8. Yes. B1 beat shuffled B2 on H2 execution (0.745798 < 0.872721) and H4 decoded (0.038513 < 0.045599).
9. Yes. B1 beat null-language B3 on H2 execution (0.745798 < 0.886908) and H4 decoded (0.038513 < 0.047128).
10. Yes. From identical current/history tensors, changing only language changed the trajectory.
11. Yes. Mean RedirectGain was **0.250126**.
12. Yes. Its session-clustered lower 95% bound was **0.136495 > 0**.
13. Yes. Execution RedirectGain was 0.183855, lower 95%=0.100917.
14. Six-way endpoint target-region accuracy was **0.516260**.
15. It was significantly above chance (lower 95%=0.492980 > 1/6), but below the frozen 0.60 macro threshold; G3 failed.
16. No on both required metrics. LCT beat prototype on H4 decoded action, but lost H2 full-latent MSE (0.691811 vs prototype 0.679007); G4 failed.
17. Current state affected endpoints descriptively, but contribution beyond language did not pass the preregistered two-metric G4 gate.
18. No. Decoded/re-encoded target identity accuracy was 0.329268, below 0.60.
19. No. Cycle error 2.800412 exceeded the development-frozen tolerance 1.434166; G5 failed.
20. No. LCT decoded-jump error exceeded direct prototype replacement; continuity gate failed.
21. The held-out `lift_blue_block_slider -> place_in_slider` case existed (7 boundaries), but the global executable-transition claim failed and the case is descriptive only.
22. Net RedirectGain was positive for 5/6 target actions; the stronger C8 multi-condition gate still failed.
23. Paraphrases preserved positive mean RedirectGain (0.248313) with low endpoint variance, but mean target accuracy was only 0.493320; robustness was partial.
24. C7 is **REJECTED**.
25. C8 is **REJECTED**.
26. Defensible claim: Language-conditioned transition was tested prospectively, but the full executable causal-redirection claim is not supported; only the individual passing components may be stated.
27. Next: a preregistered closed-loop CALVIN receding-horizon execution test with dense goal-change event labels, comparing frozen LCT against B0 and language prototype.

## Data limitation

Official CALVIN task annotations are sparse intervals rather than a dense action schedule. The next-task onset is the frozen boundary and all action frames are contiguous, but most previous/next labels have unannotated physical frames between them; this is disclosed rather than treated as a reset or silently filled.

## Defensible claim

Language-conditioned transition was tested prospectively, but the full executable causal-redirection claim is not supported; only the individual passing components may be stated.
