# -*- coding: utf-8 -*-
"""
functa_dataset package

Provides utilities for generating and loading the Functa spatial oracle dataset
used in Phase 1 of the Functa implementation plan.

The generated data consists of:
- ``X``: Tensor of shape ``(N, M, 2)`` containing sampled 2‑D coordinates.
- ``Y``: Tensor of shape ``(N, M)`` containing binary inside/outside labels.

Both tensors are stored in a single ``.pt`` file for fast torch loading.
"""
