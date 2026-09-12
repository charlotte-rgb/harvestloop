# Matched occlusion suite: `20260912_095224__s99`

- Scene seed: `99`
- Max targets: `16`
- Locked targets: `16`
- Protocol: `shared_view_rescue_matched_manifest`
- Levels: low, medium, high

## Headline: canonical success vs final success after AP

```text
canonical_success = 1 - canonical_fail_rate
final_success     = canonical_success
                  + canonical_fail_rate × rescue_rate
```

| Occlusion | N | Canonical success | Final success (AP) | Δ (pp) | Rescue |
|---|---:|---:|---:|---:|---:|
| low | 16 | 15/16 (93.8%) | 100.0% | +6.2 | 1/1 (100.0%) |
| medium | 16 | 12/16 (75.0%) | 100.0% | +25.0 | 4/4 (100.0%) |
| high | 16 | 13/16 (81.2%) | 100.0% | +18.8 | 3/3 (100.0%) |

### Claim scope

- Active perception improves **failure recovery / robustness** (rescue of bad canonical views).
- Do **not** claim monotonic improvement with occlusion.
- Do **not** claim AP improves mean localization / cut accuracy.

## Supporting metrics

| Occlusion | Canonical fail | Rescue | Views | Extra motion | Cut before | Cut after |
|---|---:|---:|---:|---:|---:|---:|
| low | 1/16 (6.2%) | 1/1 (100.0%) | 1.12 | 0.41 s | 0.0353 m | 0.0350 m |
| medium | 4/16 (25.0%) | 4/4 (100.0%) | 1.50 | 2.15 s | 0.0581 m | 0.0894 m |
| high | 3/16 (18.8%) | 3/3 (100.0%) | 1.38 | 1.72 s | 0.0978 m | 0.1049 m |

## Figures

![canonical_vs_final_success](figures/canonical_vs_final_success.png)

![canonical_failure_vs_occlusion](figures/canonical_failure_vs_occlusion.png)

![rescue_rate_vs_occlusion](figures/rescue_rate_vs_occlusion.png)

![localization_before_after](figures/localization_before_after.png)

![views_vs_occlusion](figures/views_vs_occlusion.png)

