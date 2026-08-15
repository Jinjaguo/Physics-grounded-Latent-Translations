# Offline retargeting compatibility

```json
{
  "Flow_S0_History-CFM": {
    "continuity_at_switch": 8.471840858459473,
    "distribution_shift": 0.06416171044111252,
    "only_language_changed_at_switch": true,
    "post_switch_execution_shift": 0.8850220441818237,
    "post_switch_full_shift": 1.3226845264434814
  },
  "Flow_S0_Prior-CFM": {
    "continuity_at_switch": 9.971874237060547,
    "distribution_shift": 0.08844374865293503,
    "only_language_changed_at_switch": true,
    "post_switch_execution_shift": 0.9725673794746399,
    "post_switch_full_shift": 1.506812572479248
  },
  "State_S0_RAT-C": {
    "continuity_at_switch": 9.37991714477539,
    "distribution_shift": 0.7044312953948975,
    "only_language_changed_at_switch": true,
    "post_switch_execution_shift": 2.4859745502471924,
    "post_switch_full_shift": 3.5609288215637207
  }
}
```

This changes only language after one predicted local step; it is not simulator execution.
