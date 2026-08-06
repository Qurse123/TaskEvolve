"""$0 tests for the Arm D eval driver. The subprocess runner is injected (records
argv + env), so no eval, no Together calls, no spend."""
import json

import pytest

from scripts.run_arm_d_eval import EVAL_SPLITS, run_arm_d_eval


class _CapturingRunner:
    def __init__(self):
        self.calls = []  # list of (argv, env)

    def __call__(self, argv, env):
        self.calls.append((list(argv), dict(env)))


def _run(tmp_path, **kw):
    runner = _CapturingRunner()
    index = run_arm_d_eval(
        tuned_model_id="acct/armd-inkling-lora",
        index_path=tmp_path / "index.json",
        metrics_root=tmp_path / "metrics",
        runner=runner,
        base_env={},  # deterministic, no ambient env leakage
        **kw,
    )
    return runner, index


def test_runs_both_systems_across_all_four_splits(tmp_path):
    runner, index = _run(tmp_path)
    # 2 systems x 4 splits
    assert len(runner.calls) == len(EVAL_SPLITS) * 2 == 8
    assert len(index) == 8
    assert {e["system"] for e in index} == {"base_control", "tuned"}
    # banking transfer split is present and tagged banking_knowledge
    banking = [e for e in index if e["split"] == "transfer_banking"]
    assert {e["domain"] for e in banking} == {"banking_knowledge"}
    assert len(banking) == 2  # once per system


def test_each_system_wires_its_own_model_through_together_seam(tmp_path):
    runner, index = _run(tmp_path)
    for argv, env in runner.calls:
        assert env["AGENT_API_BASE"] == "https://api.together.xyz/v1"
        assert env["HARNESS_VERSION"] == "v0.1"
        assert env["AGENT_MODEL"].startswith("openai/")
    tuned = [e for e in index if e["system"] == "tuned"]
    base = [e for e in index if e["system"] == "base_control"]
    assert all(e["agent_model"] == "openai/acct/armd-inkling-lora" for e in tuned)
    assert all(e["agent_model"] == "openai/thinkingmachines/Inkling" for e in base)


def test_uses_preregistered_seeds_and_writes_index(tmp_path):
    runner, index = _run(tmp_path)
    for argv, _env in runner.calls:
        assert "--seed-start" in argv and argv[argv.index("--seed-start") + 1] == "2001"
        assert "--repeats" in argv and argv[argv.index("--repeats") + 1] == "5"
        assert "--metrics-json" in argv  # per-repeat metrics captured
    saved = json.loads((tmp_path / "index.json").read_text())
    assert saved == index


def test_requires_a_tuned_model_id(tmp_path):
    with pytest.raises(SystemExit):
        run_arm_d_eval(tuned_model_id="", runner=_CapturingRunner(),
                       index_path=tmp_path / "i.json", metrics_root=tmp_path / "m",
                       base_env={})
