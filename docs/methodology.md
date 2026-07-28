# Cascad Methodology

Cascad studies error propagation in AI agent instances through paired
counterfactual execution traces. A trace contains node events, controlled fault
interventions, detected error observations, and native dependency edges between
components. Cascad does not claim to discover a complete structural causal
model of an arbitrary agent.

## Core Protocol

1. Define an agent graph with named nodes such as planner, memory, tool,
   verifier, and responder.
2. Run a clean control instance.
3. Run one or more perturbed instances with controlled `FaultSpec` injections.
4. Detect explicit and latent error observations.
5. Reconstruct the counterfactual propagation graph.
6. Compute propagation metrics.
7. Evaluate an intervention policy against the trace.

## Metrics

Let `G = (V, E)` be the reconstructed causal graph and let `s` be the first
fault node.

- Propagation Depth:
  `PD = max_{v in A} dist_G(s, v)`, where `A` is the affected node set.
- Propagation Delay:
  `PDelta = t(first_visible_error) - t(first_fault)`.
- Contamination Breadth Curve:
  `CB(t) = |V_affected(t)| / |V_reachable(s)|`; `CB(T)` is the terminal
  breadth scalar. The legacy static count remains only for API compatibility.
- Memory Persistence Rate:
  `MP = (# episodes following injection where poisoned memory is readable) / K`.
  This is distinct from the legacy `MAF = |O_memory| / max(1, |O_all|)`, which
  stays available only for compatibility with MVP traces.

These metrics quantify the location, extent, timing and persistence of
propagation under controlled paired-execution experiments. The validity of
broader claims depends on the evaluation family in which the metrics were
measured.

## Baselines and Ablations

Recommended experimental conditions:

- clean run without injected fault;
- planner corruption;
- memory poisoning;
- tool misresult;
- verifier false negative;
- responder-level unsafe finalization;
- intervention enabled vs disabled.

For each condition, report mean and variance over repeated trials. Cascad's
`experiment.py` module provides a small aggregation runner for these tables.

## Counterfactual reconstruction

For each corrupted trajectory, Cascad compares corresponding node events with a
clean paired trajectory. A node is marked contaminated only if its distance is
greater than its calibrated clean/clean natural divergence plus epsilon, and if
it is reachable through an already contaminated native trace dependency. The
temporal method is retained solely as an explicit ablation.

The diagnostic roles used throughout Cascad reports are operationally distinct:

- **Ground-truth source `s*`:** the node at which the scenario's controlled
  `FaultSpec` intervention is applied before execution, independently of any
  reconstruction algorithm.
- **Cascad estimate `ŝ`:** the node returned after paired event comparison,
  held-out natural-divergence thresholding, and native dependency filtering.
- **Source/root:** either `s*` when referring to experimental truth or `ŝ` when
  explicitly referring to an algorithm's estimate. Root-localization accuracy
  is `1[ŝ = s*]`.
- **Propagation mediator:** a downstream component that carries, transforms,
  amplifies, or persists contaminated state.
- **Failed safeguard:** a component with an explicit opportunity to detect or
  block contaminated state that nevertheless allowed it to continue.
- **Visible symptom:** the first component emitting an explicit externally
  observable failure.
- **Final manifestation:** the final task-level output or behavior affected by
  the cascade.

A false-negative verifier is a failed safeguard, not automatically a mediator.
For the remaining single-fault simulator protocol `s*` is scalar; future
multi-source work must use a source set.

The preferred terms are **counterfactual execution trace**, **native dependency
edge**, **counterfactual propagation graph**, and
**dependency-constrained influence path**. Broader causal language is reserved
for explicitly stated controlled-intervention assumptions.

## Statistical policy

- Wilson 95% intervals for binary proportions and accuracy;
- exact two-sided McNemar tests with complete 2×2 tables, raw discordant counts,
  and accuracy-difference effect sizes;
- deterministic bootstrap intervals for continuous or composite metrics;
- descriptive counts and intervals for small subgroups.

A non-significant paired test is not evidence of equivalence, and statistical
significance alone does not establish practical superiority.

### E-2 — Distance textuelle et correction de polarité

Pour deux textes `x` et `y`, MiniLM produit des embeddings normalisés
`e(x)` et `e(y)`. La distance sémantique brute est :

```text
D_embed(x, y) = 1 - cosine_similarity(e(x), e(y))
```

MiniLM sous-estime certaines contradictions qui ne diffèrent que par une
négation. Cascad calcule donc un indicateur `P(x, y)` qui vaut 1 lorsqu’un seul
des deux textes contient un marqueur de négation (`not`, `no`, `never`, `none`,
`without`, `cannot`, `can't`, `failed`), et 0 autrement. La distance réellement
utilisée est exactement :

```text
D_text(x, y) = max(D_embed(x, y), 0.75)  si P(x, y) = 1
D_text(x, y) = D_embed(x, y)             sinon
```

Il s’agit d’un **plancher**, et non d’un ajustement additif. La valeur reste
donc dans `[0, 1]`. La décision de contamination demeure
`D_text > mean(D_natural) + stddev(D_natural) + epsilon`.

Cette règle ne remplace pas un détecteur général de contradiction. Les
antonymes sans marqueur explicite (`approved/denied`, `eligible/ineligible`)
dépendent entièrement de MiniLM. Les deux exemples de l’audit ont été séparés
correctement (`0.412726` et `0.241695`, au-dessus d’epsilon `0.05`), mais ce
résultat ne garantit pas la couverture d’autres antonymes ou contextes.

The selected textual encoder is recorded in exported propagation metrics as
`encoder_used` and `encoder_reason`. Install `uv sync --extra semantic` to
enable the local `sentence-transformers/all-MiniLM-L6-v2` encoder.

## Threats to validity

When the MiniLM model is unavailable, Cascad uses a deterministic lexical
hashed-token fallback. It is not semantic. The audit test measured a distance of
`0.661938` for the paraphrases “the customer is eligible for a refund” / “the
client qualifies for reimbursement”, but only `0.133975` for the opposed pair
“eligible for refund” / “not eligible for refund”. Therefore fallback results
must not be used to support semantic-divergence claims in a paper; they are
acceptable only for offline smoke tests, and exports visibly declare the mode.

## Expected Paper Figures

- Agent execution graph.
- Causal propagation graph for one representative failure.
- Metric table comparing fault types.
- Ablation plot showing intervention impact on breadth/depth.
