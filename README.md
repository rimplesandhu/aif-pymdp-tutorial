# Active Inference Tutorial — NLR Internal

A hands-on introduction to Active Inference using [pymdp](https://github.com/infer-actively/pymdp).

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
uv sync
uv run jupyter notebook
```

## Notebooks

### [01 — Inference Only](notebooks/01_temp_control_inference_only.ipynb)

A temperature regulation agent that **only perceives** — no actions. The environment evolves autonomously and the agent tracks its belief over 5 hidden states (`very_cold` → `hot`) from 3 coarse, noisy observations.

- Introduces the `A` (likelihood), `B` (autonomous transitions), `C` (preferences), and `D` (initial prior) matrices
- Runs `infer_states` each step and plots true state, P(comfortable), and belief entropy
- Shows how partial observability and mean-reverting dynamics affect belief uncertainty

### [02 — Action Selection](notebooks/02_temp_control_action_selection.ipynb)

Extends notebook 01 by adding **3 control actions** (`cool`, `do_nothing`, `heat`). The agent now closes the perception–action loop using 1-step Expected Free Energy (EFE).

- `B` matrix gains an action dimension: `P(s′ | s, action)`
- Agent loop: `infer_states` → `infer_policies` → `sample_action`
- Plots true state, chosen action per step (colour-coded), and G(a) for all three actions
- Explains the pragmatic vs epistemic decomposition of EFE

## Concepts Covered

- Generative models and free energy
- Beliefs, priors, and likelihoods (`A`, `B`, `C`, `D` matrices)
- Perception as inference (state estimation)
- Action selection via expected free energy minimization

## References

- [pymdp paper](https://arxiv.org/abs/2201.03904)
- [Active Inference book](https://mitpress.mit.edu/9780262045353/active-inference/) — Parr, Pezzulo, Friston (2022)
