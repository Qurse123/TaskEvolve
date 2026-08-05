"""$0 tests for the Arm D distillation driver. The eval runner is injected (a
fake returning RepeatMetrics), so no Opus call, no eval, no real spend."""
import json

from scripts.run_distillation import DISTILL_SPLITS, run_distillation
from scripts.run_train_eval import RepeatMetrics


class _FakeRunner:
    """Records each (split, seed_start, repeats, domain) call and returns
    RepeatMetrics with deterministic run_ids so the driver's run->domain map and
    seed blocks can be asserted."""

    def __init__(self):
        self.calls = []

    def __call__(self, split, *, seed_start, repeats, domain):
        self.calls.append((split, seed_start, repeats, domain))
        return [
            RepeatMetrics(run_id=f"{split}_{seed_start + k}", seed=seed_start + k,
                          pass_rate=1.0, cost_per_successful_task=0.01)
            for k in range(repeats)
        ]


def test_maps_every_run_to_its_domain_and_writes_pointer(tmp_path):
    runner = _FakeRunner()
    pointer = tmp_path / "pointer.json"
    logs_root = tmp_path / "logs"

    run_domains = run_distillation(
        seed_start=5001, repeats=3, runner=runner,
        pointer_path=pointer, logs_root=logs_root,
    )

    # one entry per (split x repeat); domains are exactly the three trained ones
    assert len(run_domains) == len(DISTILL_SPLITS) * 3
    assert set(run_domains.values()) == {"retail", "airline", "telecom"}
    # every fake run_id maps back to its split's domain
    for (split, domain) in DISTILL_SPLITS:
        for m in runner.calls:
            if m[0] == split:
                for k in range(3):
                    assert run_domains[f"{split}_{m[1] + k}"] == domain

    saved = json.loads(pointer.read_text())
    assert saved["runs"] == run_domains
    assert saved["logs_root"] == str(logs_root)


def test_seed_blocks_are_disjoint_per_domain(tmp_path):
    runner = _FakeRunner()
    run_distillation(seed_start=5001, repeats=3, runner=runner,
                     pointer_path=tmp_path / "pointer.json", logs_root=tmp_path / "logs")

    # split i gets seeds [5001 + i*3, +3) — disjoint, contiguous blocks
    seed_starts = [c[1] for c in runner.calls]
    assert seed_starts == [5001, 5004, 5007]
    assert all(c[2] == 3 for c in runner.calls)  # repeats threaded through
    # each call's (split, domain) matches the design-fixed schedule, in order
    assert [(c[0], c[3]) for c in runner.calls] == list(DISTILL_SPLITS)


def test_only_trained_domains_never_banking():
    assert {d for _, d in DISTILL_SPLITS} == {"retail", "airline", "telecom"}
    assert "banking_knowledge" not in {d for _, d in DISTILL_SPLITS}
    assert all(s.startswith("train_distill_") for s, _ in DISTILL_SPLITS)
