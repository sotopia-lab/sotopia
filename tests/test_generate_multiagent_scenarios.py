import json
import types
from pathlib import Path
import importlib.util


def load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "examples/experimental/multi_agents_private_dm/generate_multiagent_scenarios.py"
    spec = importlib.util.spec_from_file_location("gen_scenarios", script_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return mod


def test_extract_first_json_object_variants():
    mod = load_script_module()
    obj = {"a": 1, "b": {"c": 2}}
    text1 = json.dumps(obj)
    text2 = "noise before" + text1 + "noise after"
    assert mod.extract_first_json_object(text1) == obj
    assert mod.extract_first_json_object(text2) == obj


def test_stable_hash_is_stable_and_order_independent():
    mod = load_script_module()
    a = {"x": 1, "y": {"z": 2, "w": 3}}
    b = {"y": {"w": 3, "z": 2}, "x": 1}
    assert mod.stable_hash(a) == mod.stable_hash(b)


def test_normalize_scenario_success_and_failure():
    mod = load_script_module()
    scenario = {
        "agents": [{}, {}, {}],
    }
    out = mod.normalize_scenario(scenario, sector="technology", scenario_id=7)
    assert out["scenario_id"] == 7
    assert out["sector"] == "technology"
    assert [a["agent_id"] for a in out["agents"]] == [1, 2, 3]

    bad = {"agents": [{}, {}]}  # not 3 agents
    try:
        mod.normalize_scenario(bad, sector="finance", scenario_id=1)
        assert False, "expected ValueError for non-3 agents"
    except ValueError:
        pass


def test_load_seed_pairs(tmp_path: Path):
    mod = load_script_module()
    seeds = [
        {"sector": "technology", "foo": "bar"},  # good
        {"sector": "technology", "bad": True, "reason": "bad tech"},  # bad
        {"sector": "finance", "x": 1},  # good
        {"sector": "finance", "y": 2, "reason": "bad fin"},  # bad
    ]
    p = tmp_path / "scenario_seeds.json"
    p.write_text(json.dumps(seeds), encoding="utf-8")
    pairs = mod.load_seed_pairs(str(p), ["technology", "finance"])
    assert set(pairs.keys()) == {"technology", "finance"}
    assert pairs["technology"].bad_reason == "bad tech"


def test_run_sector_with_mock_llm():
    mod = load_script_module()

    # Ensure API key check won't fail elsewhere if invoked
    mod.OPENAI_API_KEY = "test-key"

    # Prepare seed pair
    seed_pair = mod.SeedPair(good={"good": True}, bad={"bad": True, "reason": "why"}, bad_reason="why")

    # Mock llm_call_text to return deterministic JSON strings
    def fake_llm_call_text(client, model, system, user, effort, max_retries=6, tag=""):
        if "GEN" in tag:
            # Return a candidate with 3 agents
            return json.dumps({
                "agents": [{"role": "A"}, {"role": "B"}, {"role": "C"}]
            })
        else:
            # Judge: mark correct and echo corrected_scenario_json
            return json.dumps({
                "is_logically_correct": True,
                "anomalies_reason": "",
                "corrected_scenario_json": {
                    "agents": [{"role": "A"}, {"role": "B"}, {"role": "C"}],
                    "extra": "fixed"
                }
            })

    orig = mod.llm_call_text
    mod.llm_call_text = fake_llm_call_text  # type: ignore
    try:
        sector, accepted, counts, audit = mod.run_sector(
            client=types.SimpleNamespace(),
            sector="technology",
            seed_pair=seed_pair,
            target=1,
            max_attempts=3,
            number=3,
            gen_model="dummy",
            gen_effort="low",
            judge_model="dummy",
            judge_effort="low",
        )
        assert sector == "technology"
        assert len(accepted) == 1
        assert counts["accepted"] == 1
        # audit should contain corrected scenario, no wrong original since marked correct
        assert any(isinstance(x, dict) and x.get("extra") == "fixed" for x in audit)
    finally:
        mod.llm_call_text = orig  # type: ignore


def test_run_sector_with_wrong_case_includes_reason():
    mod = load_script_module()
    mod.OPENAI_API_KEY = "test-key"
    seed_pair = mod.SeedPair(good={}, bad={"reason": "bad"}, bad_reason="bad")

    def fake_llm_call_text(client, model, system, user, effort, max_retries=6, tag=""):
        if "GEN" in tag:
            return json.dumps({
                "agents": [{}, {}, {}],
                "foo": "orig"
            })
        else:
            return json.dumps({
                "is_logically_correct": False,
                "anomalies_reason": "X failed",
                "corrected_scenario_json": {"agents": [{}, {}, {}], "bar": "corr"}
            })

    orig = mod.llm_call_text
    mod.llm_call_text = fake_llm_call_text  # type: ignore
    try:
        _, accepted, counts, audit = mod.run_sector(
            client=types.SimpleNamespace(),
            sector="finance",
            seed_pair=seed_pair,
            target=1,
            max_attempts=2,
            number=3,
            gen_model="dummy",
            gen_effort="low",
            judge_model="dummy",
            judge_effort="low",
        )
        assert len(accepted) == 1
        # audit should contain 2 entries: corrected and wrong with reason
        reasons = [x.get("reason") for x in audit if isinstance(x, dict)]
        assert any(r and r.startswith("Bad seed: X failed") for r in reasons)
    finally:
        mod.llm_call_text = orig  # type: ignore

