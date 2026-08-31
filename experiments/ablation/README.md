# V3 Process-Verifiable Ablations

All variants share one fixed cohort, NCDM checkpoint, LLM configuration,
target ordering, feedback mode, and scoring path.

- `full`: all four V3 modules.
- `no-evidence-representation`: remove provenance-bearing history evidence;
  retain NCDM/IRT state.
- `no-state-item-alignment`: retain bounded recent evidence but remove
  item-conditioned relevance and explicit alignment.
- `no-structured-response-process`: replace the eight-field process contract
  with the direct three-field response contract.
- `no-dynamic-state-evolution`: freeze prompt-facing state over target steps.
- `direct-response-generation`: remove evidence representation, alignment,
  structured process, and state evolution.

Every variant makes at most one API call per target item. Full raw prompts,
responses, selected evidence, source IDs, alignment traces, process consistency,
and state transitions are archived for offline analysis.
