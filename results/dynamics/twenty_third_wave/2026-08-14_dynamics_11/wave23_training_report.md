# Wave 23 training report

Only exact Wave21 B1 LCT parameters were trainable. Encoder, decoder, text projection, goal cores, and neighborhood candidates were frozen. The only new term was state-conditioned K=20 goal-core softmin alignment.

| lambda | seed | final loss | gradient audit |
|---:|---:|---:|---|
| 0.03 | 210821 | 0.97730488 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.03 | 210822 | 0.95808944 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.03 | 210823 | 0.96564636 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.03 | 210824 | 0.97415064 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.03 | 210825 | 0.96133981 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.03 | 210826 | 1.01237785 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.1 | 210821 | 1.03876683 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.1 | 210822 | 1.02120106 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.1 | 210823 | 1.02986434 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.1 | 210824 | 1.03769513 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.1 | 210825 | 1.02628605 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.1 | 210826 | 1.07600807 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.3 | 210821 | 1.19052774 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.3 | 210822 | 1.17796460 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.3 | 210823 | 1.18532304 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.3 | 210824 | 1.20551508 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.3 | 210825 | 1.19888120 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |
| 0.3 | 210826 | 1.22432220 | {'transition_gradients_nonzero': True, 'representation_gradients_none': True, 'neighbor_requires_grad': False, 'classification_loss': 0.0, 'prototype_loss': 0.0, 'cycle_loss': 0.0} |

Development selection status: **NO_CANDIDATE_PASSED**; selected lambda=None.
