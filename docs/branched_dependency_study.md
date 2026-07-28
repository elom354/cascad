# Native Dependency-Constraint Study

## Objective

This simulator-level study isolates the contribution of native dependency
constraints. It was specified before evaluation and does not modify the
previous `cloud_distant_symptom` or natural-noise studies.

The fixed topology is:

```text
planner
   ├── share -> memory -> notify -> responder
   └── audit_log -> metrics_export
```

There is no native edge or dependency path from `share`, `memory`, or `notify`
to the independent branch. The diagnostic roles remain:

- root: `share`;
- mediator: `memory`;
- first visible symptom: `notify`;
- true contaminated subgraph: `share`, `memory`, `notify`, `responder`.

`audit_log` and `metrics_export` are never ground-truth contaminated.

## Frozen divergence classes

The clean/clean calibration split contains 24 pairs and declares only
`metrics_export.format_hint` naturally variable.

The 80 held-out evaluation pairs contain 20 instances per level:

- B0: no independent divergence;
- B1: one held-out `format_hint` variation at `metrics_export`, within the
  calibrated range;
- B2: an above-threshold audit destination change at `audit_log`;
- B3: above-threshold destination and aggregation-scope changes at both
  independent nodes, with their events temporally interleaved with the main
  branch.

The B2/B3 values and rationales are exported in the evaluation manifest as the
pre-evaluation specification. They contain no explicit evaluative wording and
are not read by the main workflow. Removing the independent branch leaves both
the clean and corrupt main outcomes unchanged.

## Methods and metric

All methods receive the same trace pairs:

1. temporal adjacency;
2. naive first raw difference;
3. maximum raw divergence;
4. Cascad without calibration, with dependencies;
5. Cascad with calibration, without dependencies;
6. full Cascad with calibration and native dependencies.

The primary metric is:

```text
IBFIR =
  divergent independent-branch nodes marked contaminated
  / divergent independent-branch nodes
```

## Command

```bash
uv run --extra semantic python -m cascad branched-dependency-study \
  --instances-per-level 20 \
  --calibration-pairs 24 \
  --out runs/branched-dependency-study
```

## Results

All 80 instance IDs, clean trace hashes, corrupt trace hashes, and optional
paired prompt hashes are unique. The split, topology, threshold-class, leakage,
and branch-removal audits passed with the MiniLM embedding encoder.

| Method | B0 root / IBFIR | B1 root / IBFIR | B2 root / IBFIR | B3 root / IBFIR |
|---|---:|---:|---:|---:|
| temporal adjacency | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 1.00 | 0.00 / 1.00 |
| naive first raw difference | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.00 / 0.50 |
| maximum raw divergence | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Cascad without calibration | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |
| Cascad without dependencies | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 1.00 | 0.00 / 1.00 |
| full Cascad | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |

At B2, the no-dependency ablation included `audit_log` in every run and had
subgraph precision `0.80`; full Cascad excluded it and retained precision and
recall `1.00`. At B3, the ablation included both independent nodes in every run,
had precision `0.667`, and selected the temporally earlier `audit_log`; full
Cascad again recovered only the four true main-branch nodes.

The no-calibration condition also excluded the independent branch because its
native dependency filter was active. B1 confirms independently that calibration
absorbs the ordinary held-out variation.

## Artifacts

`runs/branched-dependency-study/` contains:

- calibration and evaluation manifests;
- fairness and exact graph-topology audits;
- 480 raw JSON/JSONL results;
- CSV/JSON summaries;
- paired comparisons;
- deterministic bootstrap intervals;
- root-accuracy, IBFIR, precision, and recall SVG plots.

## Bounded conclusion

Native dependency constraints reduced false inclusion of salient disconnected
divergences in this 80-instance branched simulator family. They preserved the
complete contaminated main branch while reducing IBFIR from `1.00` to `0.00`
at B2 and B3. This result does not establish generalization to arbitrary agent
graphs, real tools, or uncontrolled production traces.
