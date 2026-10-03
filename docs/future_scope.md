# Future scope

Ideas that are decided to be worth doing later but are NOT started. Nothing here has results yet.

---

## 1. Gradient descent as the teacher for the per-feature network (FeatureNet)

**Status:** idea only. Not implemented, no results.

### The situation it comes from
- Per-feature multipliers tuned directly on a prompt by gradient descent (the "oracle" in `src/feature_models.py`, and the Gradient descent option in the Hybrid tab, `src/gradient_editing.py`) reach rank 1 far more often than anything learned so far. In the one-shot proxy (Entry 27) the free per-feature optimum reached about 73% of held-out prompts, against about 31 to 32% for the trained FeatureNet and about 37% for the single best mute/boost pair.
- The network is the fast version of the same idea (one small pass instead of about 100 gradient steps), but in the development runs it captured little of that gain. It was trained from scratch to minimise the loss, which was hard to optimise and sensitive to the loss weights.

### The idea
Use the gradient-descent multipliers as labels. For each training prompt, run the per-prompt gradient descent and keep the multipliers it ends with. Then train FeatureNet to predict them from the same inputs it uses now (the 15 per-candidate descriptors plus the 12 prompt summary numbers). The network no longer has to discover good multipliers by itself; it only has to imitate multipliers already known to work.

### Plan
1. Generate labels: run the per-prompt gradient descent on the training prompts (about 1,000 for the first test) and store the final multiplier of every candidate. The app measured about 5 seconds per prompt (100 steps, Top N 200, GTX 1660 Ti), which would be about 1.4 hours for 1,000 prompts; the batched version in `oracle_feature` (cached layer-8 states) is probably faster, not measured.
2. Train FeatureNet by regression on those multipliers (start with plain mean squared error, then try a version that matches the effect on the output logits instead of the raw multipliers).
3. Evaluate on the same held-out prompts and the same one-shot proxy, with the same paired tests (exact McNemar) against: the version 1 pair, the trained-from-scratch FeatureNet, and the free per-feature optimum (the teacher itself, which is the ceiling).
4. Only if it holds up in the proxy, test the network's multipliers in the real sweep.

### Known risks
- Gradient descent has many equally good solutions (different multipliers can give the same rank), so the labels are noisy and a network copying them may learn an average that works worse than any single solution. Matching effects rather than raw multipliers, or tuning the teacher with a pull toward a common starting point, are possible fixes.
- The teacher's side effects follow its loss weights (see the failing-prompt note below); a network copying it inherits them.
- Whether the gain survives outside the one-shot proxy is untested for any per-feature rule.

### What this would buy
Roughly the teacher's success rate at the cost of one small pass instead of about 100 gradient steps per prompt. If the network cannot get close to the teacher, the plain per-prompt gradient descent stays the method and the network is dropped.

### Related note (not part of this idea)
The gradient-descent loss currently penalises side effects (KL) only once the target already leads, so a prompt that fails can end with a heavily distorted output (KL 2.7 on the "The sky is" to "Adi" test). Anything trained on its labels should be checked for this first.

---

Context for item 1: `docs/Research_Journal/28.md` (the gradient-descent method in the Hybrid tab and the hand trials).
