# CCDR Pilot: Calibration-Constrained Divergent Rounding
## A Blind, Model-Agnostic PTQ Pilot for Fingerprint Removal

## 1. Objective

We want a post-training quantization method that can reduce/remove a hidden fingerprint without using:

- the clean/base model;
- fingerprint keys, triggers, prompts, or responses;
- the fingerprint verifier;
- knowledge of the fingerprint embedding algorithm;
- manual localization of important blocks;
- fine-tuning or QAT.

The method receives only:

```text
1. a fingerprinted FP model;
2. generic calibration data;
3. the same INT4 quantization format used by RTN4.
```

The method should be applied in exactly the same way to every model.

The working hypothesis is:

> Standard RTN preserves the fingerprint because it explicitly chooses the quantized model that stays closest to the fingerprinted FP weights. We instead search for a different INT4 rounding configuration that moves farther from the fingerprinted model while preserving ordinary behavior on generic calibration data.

Temporary method name:

```text
CCDR = Calibration-Constrained Divergent Rounding
```

---

## 2. Core Principle

For one linear layer with FP weight

\[
W \in \mathbb{R}^{d_{out}\times d_{in}}
\]

and calibration input activations

\[
X \in \mathbb{R}^{N\times d_{in}},
\]

standard RTN gives

\[
Q_0 = Q_{\mathrm{RTN4}}(W).
\]

Normal FP output:

\[
Y = XW^T.
\]

RTN4 output:

\[
Y_0 = XQ_0^T.
\]

RTN reconstruction error:

\[
E_0 = \frac{1}{N d_{out}}\|Y_0-Y\|_F^2.
\]

We want another valid INT4 solution \(Q\) that is farther from the fingerprinted FP weight while keeping normal reconstruction close to RTN4:

\[
\boxed{\max_Q \|Q-W\|_F^2}
\]

subject to

\[
\boxed{E(Q) \le (1+\epsilon)E_{\mathrm{RTN4}}.}
\]

The first pilot does not solve this combinatorial optimization exactly. It uses the structured search below.

---

## 3. Keep the Existing RTN4 Grid Fixed

For the first implementation:

- compute RTN4 scale exactly as before;
- compute RTN4 zero-point exactly as before;
- keep scale and zero-point fixed;
- do not optimize clipping;
- do not optimize scale;
- do not optimize zero-point;
- change only rounding decisions.

This isolates one question:

> Can a different rounding configuration within the same INT4 grid move farther from the fingerprinted model while preserving generic behavior?

---

## 4. Standard vs Opposite Adjacent Rounding

For a scalar weight \(w\), let the current asymmetric quantizer have scale \(s\) and zero-point \(z\).

Use the exact pre-round expression from the existing RTN code. Conceptually:

\[
u = w/s + z.
\]

Define

\[
q_{lo}=\lfloor u\rfloor,
\qquad
q_{hi}=\lceil u\rceil.
\]

RTN chooses the nearest valid integer level:

\[
q_{rtn}=\mathrm{round}(u).
\]

Define the opposite adjacent level:

```python
if q_rtn == q_lo:
    q_alt = q_hi
else:
    q_alt = q_lo
```

Important rules:

- clamp/check against the valid INT4 code range;
- if `q_lo == q_hi`, there is no useful alternative;
- if the alternative is invalid, keep RTN;
- do not jump by two or more bins in V1;
- dequantize with the existing RTN scale/zero-point convention exactly.

---

## 5. Use Generic Calibration Only to Protect Normal Behavior

We do not know where the fingerprint is stored.

Calibration is therefore **not** used to identify fingerprint weights. It is only used to identify which input directions normal text uses strongly or weakly.

For each input channel \(j\) of a linear layer, compute activation energy:

\[
a_j = \frac{1}{N}\sum_{t=1}^N X_{t,j}^2.
\]

Interpretation:

```text
large a_j  -> generic data strongly uses this direction
small a_j  -> generic data weakly uses this direction
```

CCDR first perturbs low-energy directions.

The hypothesis is:

> Move farther in directions weakly constrained by normal calibration data while preserving directions strongly required by normal behavior.

This rule is model-agnostic and does not depend on block index or fingerprint knowledge.

---

## 6. Candidate Construction

The current quantizer uses:

```text
group_size = 128
per-output-row / per-input-group
```

For each contiguous group of 128 input columns:

```text
group g = input columns [128*g : 128*(g+1)]
```

compute the 128 input-channel energies and sort them from lowest to highest.

Use nested candidate sizes:

```text
K = [0, 1, 2, 4, 8, 16, 32]
```

Meaning:

```text
k = 0:
    exactly standard RTN4

k = 1:
    in every 128-input group, use opposite rounding for the
    single lowest-energy input channel

k = 2:
    flip the two lowest-energy input channels

k = 4:
    flip the four lowest-energy input channels

...
```

For one tensor, selected input positions are shared across output rows because the ranking comes from the common input activation channels. Each scalar weight still uses its own valid `q_alt`.

This makes the first pilot simple and vectorizable.

---

## 7. Why This Search Is General

The candidates are nested:

\[
Q_0,Q_1,Q_2,Q_4,Q_8,Q_{16},Q_{32}.
\]

As more valid opposite-rounding decisions are used, the quantized tensor generally moves farther from the FP tensor.

The reconstruction constraint decides how aggressive each layer can be:

```text
sensitive layer   -> may select k = 0 or 1
insensitive layer -> may select k = 8, 16, or 32
```

There is no rule such as:

```text
modify blocks 20-27
```

Every layer uses the same algorithm.

---

## 8. Layer-Level Candidate Selection

For each candidate \(k\), construct the full quantized tensor \(Q_k\).

Compute:

\[
Y_k = XQ_k^T
\]

and reconstruction error:

\[
E_k = \frac{1}{N d_{out}}\|Y_k-XW^T\|_F^2.
\]

Also compute parameter divergence:

\[
D_k = \frac{1}{d_{out}d_{in}}\|Q_k-W\|_F^2.
\]

Baseline:

\[
E_0 = E_{k=0}.
\]

A candidate is feasible if:

\[
E_k \le (1+\epsilon)E_0.
\]

Among feasible candidates, select:

\[
k^* = \arg\max_k D_k.
\]

If no alternative candidate is feasible, use:

```text
k* = 0
```

and keep standard RTN4 for that tensor.

---

## 9. First Pilot Configuration

Use only:

```text
bits = 4
group_size = 128
K = [0, 1, 2, 4, 8, 16, 32]
epsilon = 0.05
calibration_tokens = 1024
```

`epsilon = 0.05` means the layer reconstruction MSE may be at most 5% larger than RTN4 reconstruction MSE.

This is **not** a 5% allowed drop in downstream utility.

Do not introduce extra tuning parameters in the first run.

Only after implementation is verified, optional epsilon ablation can use:

```text
0.00, 0.02, 0.05, 0.10
```

---

## 10. Calibration Data

Use generic text only, preferably the same generic calibration source already used in the repository, e.g. C4.

Do not use:

```text
fingerprint prompts
fingerprint responses
secret fingerprint keys
fingerprint verifier feedback
base model
```

For the first fast pilot, use about 1024 token activation vectors per target module.

---

## 11. Target Modules

Match the current RTN4 tensor discovery exactly:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Keep the existing excluded modules excluded.

Do not add `lm_head` just for CCDR.

---

## 12. Activation Collection

For each target linear module, obtain its real input activations:

```text
X: [num_calibration_tokens, in_features]
```

Reuse existing calibration hooks/cache if available.

For memory efficiency:

- store activations on CPU if needed;
- store FP16/BF16 and convert to FP32 for statistics if practical;
- process one module/tensor at a time where possible;
- release candidate tensors after selection;
- do not keep all target-module activations on GPU simultaneously.

---

## 13. Required Code Functions

### 13.1 Expose RTN state

Reuse the current RTN4 implementation and expose its internal state:

```python
def rtn4_with_state(weight, group_size=128):
    # return:
    # q_rtn_int
    # dequant_rtn
    # scale
    # zero_point
    # pre_round_code
    ...
```

Do not change RTN numerics.

Critical test:

```text
CCDR with K=[0] must exactly reproduce the current RTN4 model.
```

---

### 13.2 Build opposite codes

```python
def build_opposite_codes(pre_round_code, q_rtn_int, qmin, qmax):
    u = pre_round_code

    q_lo = torch.floor(u)
    q_hi = torch.ceil(u)

    q_alt = torch.where(
        q_rtn_int == q_lo,
        q_hi,
        q_lo,
    )

    valid = (
        (q_alt >= qmin)
        & (q_alt <= qmax)
        & (q_lo != q_hi)
    )

    q_alt = torch.where(valid, q_alt, q_rtn_int)
    return q_alt
```

Adapt dtype/code conventions to the existing quantizer.

---

### 13.3 Compute channel energy

```python
def compute_channel_energy(X):
    # X: [N, in_features]
    return X.float().pow(2).mean(dim=0)
```

---

### 13.4 Rank channels inside each 128-input group

```python
def rank_channels_per_group(energy, group_size=128):
    rankings = []
    for start in range(0, energy.numel(), group_size):
        end = min(start + group_size, energy.numel())
        local = energy[start:end]
        order = torch.argsort(local, descending=False)
        rankings.append(order)
    return rankings
```

Handle the last short group correctly if `in_features` is not divisible by 128.

---

### 13.5 Build one candidate

```python
def build_candidate(
    q_rtn_int,
    q_alt_int,
    scale,
    zero_point,
    group_rankings,
    k,
    group_size=128,
):
    # Start from RTN codes.
    q = q_rtn_int.clone()

    # For every input group, choose the k lowest-energy input positions.
    # Apply those input-column positions to all output rows.
    # Where a valid alternative exists, replace q_rtn with q_alt.

    # Finally dequantize using the ORIGINAL RTN scale/zero-point.
    return q_dequant
```

Do not flatten the whole tensor and take a global top-k.

`k` is **per 128-input group**.

---

## 14. Candidate Search Function

Implement:

```python
@torch.no_grad()
def choose_ccdr_candidate(
    W,
    X,
    rtn_state,
    k_values=(0, 1, 2, 4, 8, 16, 32),
    epsilon=0.05,
):
    ...
```

Reference pseudo-code:

```python
Y_fp = X @ W.T

Q0 = rtn_state.dequant_weight
Y0 = X @ Q0.T

E0 = mse(Y0, Y_fp)
D0 = mean((Q0 - W) ** 2)

energy = compute_channel_energy(X)
rankings = rank_channels_per_group(energy)

best_Q = Q0
best_k = 0
best_E = E0
best_D = D0

for k in k_values[1:]:
    Qk = build_candidate(
        q_rtn_int=rtn_state.q_int,
        q_alt_int=rtn_state.q_alt_int,
        scale=rtn_state.scale,
        zero_point=rtn_state.zero_point,
        group_rankings=rankings,
        k=k,
    )

    Yk = X @ Qk.T

    Ek = mse(Yk, Y_fp)
    Dk = mean((Qk - W) ** 2)

    feasible = Ek <= (1.0 + epsilon) * E0

    if feasible and Dk > best_D:
        best_Q = Qk
        best_k = k
        best_E = Ek
        best_D = Dk

return best_Q, {
    "selected_k": best_k,
    "rtn_error": E0,
    "selected_error": best_E,
    "rtn_drift": D0,
    "selected_drift": best_D,
}
```

No gradients are required.

Use `torch.no_grad()` throughout the quantization/search path.

---

## 15. Numerical Consistency

For correctness testing, compute reconstruction in FP32 where practical:

```python
Y_fp = torch.nn.functional.linear(X.float(), W.float())
Y_q  = torch.nn.functional.linear(X.float(), Q.float())
```

If this is too expensive, BF16/FP16 GEMM is acceptable, but accumulate/compare the final squared errors in FP32 consistently.

Do not calculate RTN error and CCDR error with different numerical precision.

---

## 16. Speed Optimization — Only After Correctness

The naive pilot computes a GEMM for every candidate.

First make that version correct.

Later optimize using:

\[
Y_k = Y_0 + X(Q_k-Q_0)^T.
\]

Because `Q_k-Q_0` affects only selected input columns, this can be cheaper.

Do not optimize before the one-tensor test is correct.

---

## 17. Model-Level Quantization Procedure

Pseudo-code:

```python
model = load_fingerprinted_model()
calib = load_generic_calibration_data()

for module_name, linear in target_linear_modules(model):
    X = get_calibration_inputs(module_name)
    W = linear.weight.data

    rtn_state = rtn4_with_state(
        W,
        group_size=128,
    )

    rtn_state.q_alt_int = build_opposite_codes(
        rtn_state.pre_round_code,
        rtn_state.q_int,
        rtn_state.qmin,
        rtn_state.qmax,
    )

    Q_selected, stats = choose_ccdr_candidate(
        W=W,
        X=X,
        rtn_state=rtn_state,
        k_values=[0, 1, 2, 4, 8, 16, 32],
        epsilon=0.05,
    )

    linear.weight.data.copy_(Q_selected)
    save_stats(module_name, stats)

save_quantized_model(...)
```

Do not quantize modules that standard RTN4 excluded.

---

## 18. Required Logging

Create:

```text
results/ccdr_layer_stats.csv
```

One row per quantized tensor.

Columns:

```text
tensor_name
block_id
module_type
in_features
out_features
selected_k
selected_flip_fraction
rtn_reconstruction_mse
ccdr_reconstruction_mse
reconstruction_ratio
rtn_weight_mse
ccdr_weight_mse
weight_drift_ratio
```

Definitions:

```text
selected_flip_fraction = selected_k / group_size
reconstruction_ratio   = ccdr_reconstruction_mse / rtn_reconstruction_mse
weight_drift_ratio      = ccdr_weight_mse / rtn_weight_mse
```

Expected for each tensor:

```text
reconstruction_ratio <= 1 + epsilon
```

The main diagnostic is:

```text
weight_drift_ratio > 1
```

while reconstruction remains near RTN4.

---

## 19. Sanity Tests Before Full Model Evaluation

### Test 1 — `k=0` reproduces RTN4

Force:

```text
K = [0]
```

The resulting model must reproduce the current RTN4 baseline.

Check:

```text
same quantized tensors
same numerical weights
same fingerprint result
same WikiText-2 PPL (up to normal numerical tolerance)
```

Do not proceed if this fails.

### Test 2 — inspect candidate trade-off for a few tensors

Print:

```text
k
reconstruction_error / RTN_error
weight_distance / RTN_weight_distance
```

This confirms that candidates genuinely move farther from FP weights.

### Test 3 — quantizer has no fingerprint access

The CCDR quantization path must never load:

```text
fingerprint dataset
fingerprint target
fingerprint verifier
base model
```

Fingerprint data is used only after quantization for evaluation.

---

## 20. First Experiment

Evaluate only three models first:

```text
1. IF-FP
2. IF-RTN4
3. IF-CCDR4
```

Fingerprint evaluation:

```text
official IF verified / 8
generated outputs
```

Utility:

```text
WikiText-2 perplexity
same lightweight utility tasks already available in the repository
```

Do not run every fingerprint method before the pilot is understood.

---

## 21. What Counts as a Promising Result

Desired qualitative pattern:

```text
IF-FP:
    fingerprint = 8/8

IF-RTN4:
    fingerprint = 8/8
    utility good

IF-CCDR4:
    fingerprint clearly lower
    utility close to RTN4
```

The first run does not have to reach `0/8`.

A clear fingerprint reduction at similar utility is enough to justify improving the solver.

---

## 22. Failure Modes

### A. Almost every layer selects `k=0`

First check implementation.

If correct, try:

```text
epsilon = 0.10
```

before changing the method.

### B. Utility collapses

Check:

- activation inputs belong to the correct module;
- RTN scale/zero-point are reused correctly;
- opposite rounding is correct;
- the final model is not accidentally quantized twice;
- reconstruction error is evaluated correctly.

If implementation is correct, reduce `epsilon`.

### C. Utility remains good but fingerprint stays `8/8`

This means the V1 low-energy structured heuristic is insufficient.

Do not add fingerprint-aware selection.

Keep the same general constrained-divergence principle and move to the stronger V2 solver below.

---

## 23. V2 — Residual-Aware Individual Rounding Search

Do not implement V2 before testing V1.

For one output row, let current RTN reconstruction residual be:

\[
e = X(q-w).
\]

If one quantized coordinate at input channel \(j\) changes by \(\delta\), the squared reconstruction-error change is exactly:

\[
\Delta E
=
2\delta e^T X_{:,j}
+
\delta^2\|X_{:,j}\|_2^2.
\]

This allows a stronger solver to choose individual opposite-rounding flips that:

```text
increase parameter divergence
while consuming as little generic reconstruction budget as possible
```

This is more expensive but still blind and model-agnostic.

---

## 24. Why the Method Is General

The algorithm contains no reference to:

```text
IF-SFT
specific fingerprint tokens
specific fingerprint prompts
specific block ranges
base checkpoint
fingerprint verifier
```

The same code can be applied to different fingerprinted models.

Only after the IF-SFT pilot is promising should the exact same CCDR configuration be evaluated on additional fingerprint methods.

Do not retune block positions for each fingerprint method.

---

## 25. Generalization Test

If IF-SFT shows a useful trade-off, reuse exactly:

```text
bits = 4
group_size = 128
K = [0, 1, 2, 4, 8, 16, 32]
epsilon = 0.05
same generic calibration protocol
```

on the other fingerprinted checkpoints.

Then evaluate each method using its own official verifier.

A generalization claim requires the same blind quantizer to reduce multiple fingerprint types without model-specific localization.

---

## 26. Suggested CLI

```bash
python run_ccdr.py \
  --model <fingerprinted-model> \
  --bits 4 \
  --group-size 128 \
  --calib-data c4 \
  --calib-tokens 1024 \
  --epsilon 0.05 \
  --k-values 0 1 2 4 8 16 32 \
  --output-dir outputs/ccdr_if
```

Run fingerprint evaluation separately:

```bash
python eval_fingerprint.py \
  --model outputs/ccdr_if \
  --fingerprint-data Model-Fingerprint/dataset/llama_fingerprint_chat
```

Run utility evaluation separately:

```bash
python eval_ppl.py \
  --model outputs/ccdr_if \
  --dataset wikitext2
```

The quantizer itself must not see fingerprint evaluation data.

---

## 27. Minimal Implementation Order

Implement in this order:

```text
1. Expose RTN integer codes, scale, zero-point, and pre-round values.
2. Verify K=[0] exactly reproduces current RTN4.
3. Implement opposite adjacent rounding.
4. Capture generic calibration activations for one linear tensor.
5. Compute per-input-channel activation energy.
6. Build K candidates for one tensor.
7. Compute reconstruction error and parameter divergence.
8. Select the farthest feasible candidate.
9. Verify everything on one tensor.
10. Apply to all current RTN4 target tensors.
11. Save layer statistics.
12. Evaluate IF fingerprint.
13. Evaluate WikiText-2 PPL.
```

Do not optimize runtime before steps 1-9 are correct.

---

## 28. Short Instruction for Codex

Implement a blind INT4 PTQ variant called CCDR on top of the existing asymmetric RTN4 implementation.

For every quantized linear tensor:

1. collect generic calibration inputs;
2. compute standard RTN4 and preserve its scale/zero-point;
3. construct the other adjacent rounding level for each scalar weight;
4. rank input channels inside each 128-input group by generic activation energy;
5. build candidates by applying opposite rounding to the lowest-energy `k` channels for `k = [0,1,2,4,8,16,32]`;
6. evaluate candidate layer-output reconstruction error;
7. keep only candidates with error no larger than `(1+epsilon) * RTN_error`;
8. among feasible candidates choose the one farthest from the fingerprinted FP weight;
9. apply exactly the same rule to every layer;
10. never use the base model, fingerprint dataset, fingerprint verifier, or block-specific rules during quantization.

First pilot:

```text
bits = 4
group_size = 128
calibration_tokens = 1024
epsilon = 0.05
```

Primary research question:

> Can we choose an INT4 model that moves farther from the fingerprinted FP parameters than RTN4 while preserving generic layer behavior, and thereby reduce hidden fingerprint verification?
