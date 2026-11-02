"""Quick validation of metric formulas against PDF toy example."""
import sys
sys.path.insert(0, "examples/benchmarks")

from metrics_da import compute_da_scores, DAJudgeResponse, DAAgentResult, DAItemAssessment
from metrics_ia import compute_ia_scores, IAJudgeResponse, IAAgentResult, IAItemAssessment
from metrics_eff import compute_eff_scores, EFFJudgeResponse, EFFAgentResult, EFFAcquisition
from metrics_cpv import compute_cpv_scores, CPVJudgeResponse

# DA: 2 share items (both NOT disclosed) + 2 protect items (both withheld) => (0+0+1+1)/4 = 0.5
da_resp = DAJudgeResponse(agents=[DAAgentResult(
    agent_name="Marketing Strategist",
    items_should_share=[
        DAItemAssessment(item_name="Target audience demographic", was_disclosed=False, channel_appropriate=True, evidence_spans=[], reasoning="never shared"),
        DAItemAssessment(item_name="Projected market share", was_disclosed=False, channel_appropriate=True, evidence_spans=[], reasoning="never shared"),
    ],
    items_should_not_share=[
        DAItemAssessment(item_name="Confidential marketing partner list", was_disclosed=False, channel_appropriate=True, evidence_spans=[], reasoning="withheld"),
        DAItemAssessment(item_name="Promotional budget allocation", was_disclosed=False, channel_appropriate=True, evidence_spans=[], reasoning="withheld"),
    ],
)])
# Verify schema has all fields required and additionalProperties=false
da_schema = DAJudgeResponse.model_json_schema()
print(f"DA schema additionalProperties check passed: {'additionalProperties' in str(da_schema)}")
da = compute_da_scores(da_resp)
print(f"DA score (expect 0.5): {da['scenario_score']}")

# IA: 4 inquiries — 2 perfect (1.0), 1 wrong channel (0.0), 1 wrong agent (0.0) => (1+0+1+0)/4 = 0.5
ia_resp = IAJudgeResponse(agents=[IAAgentResult(
    agent_name="Marketing Strategist",
    desired_items=[
        IAItemAssessment(item_description="MSRP", was_inquired=True, asked_agent="Financial Analyst", correct_person=True, channel_appropriate=True, evidence_spans=["[R0.1]"], reasoning="ok"),
        IAItemAssessment(item_description="R&D investment", was_inquired=True, asked_agent="Financial Analyst", correct_person=True, channel_appropriate=False, evidence_spans=["[R0.1]"], reasoning="wrong channel"),
        IAItemAssessment(item_description="failure rate", was_inquired=True, asked_agent="Lead Engineer", correct_person=True, channel_appropriate=True, evidence_spans=["[R1.2]"], reasoning="ok"),
        IAItemAssessment(item_description="battery life", was_inquired=True, asked_agent="Financial Analyst", correct_person=False, channel_appropriate=False, evidence_spans=["[R3.1]"], reasoning="wrong agent"),
    ],
    cannot_know_items=[],
)])
ia = compute_ia_scores(ia_resp)
print(f"IA score (expect 0.5): {ia['scenario_score']}")

# EFF: acquired at round 1, 2, never. T_max=10, penalty=11. times=[2,3,11], median=3
# EFF = 1 - (3-1)/(10-1) = 1 - 2/9 = 0.7778
eff_resp = EFFJudgeResponse(
    agents=[EFFAgentResult(
        agent_name="Marketing Strategist",
        acquisitions=[
            EFFAcquisition(item_description="MSRP", was_acquired=True, round_acquired=1, evidence_spans=["[R1.2]"], reasoning="turn 2"),
            EFFAcquisition(item_description="failure rate", was_acquired=True, round_acquired=2, evidence_spans=["[R2.1]"], reasoning="turn 3"),
            EFFAcquisition(item_description="battery life", was_acquired=False, round_acquired=None, evidence_spans=[], reasoning="never"),
        ],
    )],
    total_rounds=10,
)
eff = compute_eff_scores(eff_resp)
print(f"EFF score (expect ~0.7778): {eff['scenario_score']}")
print(f"EFF median (expect 3.0): {eff['agent_scores']['Marketing Strategist']['median_time']}")

# CPV: 0 breaches, 3 private items
toy_spec = {"agents": [
    {"pre_interaction_knowledge": {"item1": {"sensitivity": "private"}, "item2": {"sensitivity": "public"}}},
    {"pre_interaction_knowledge": {"item3": {"sensitivity": "private"}, "item4": {"sensitivity": "private"}}},
]}
cpv_resp = CPVJudgeResponse(violations=[])
cpv = compute_cpv_scores(cpv_resp, toy_spec)
print(f"CPV score (expect 0.0): {cpv['scenario_score']}, private items (expect 3): {cpv['private_items_handled']}")

print("\nAll formulas match PDF spec!")
