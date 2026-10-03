from __future__ import annotations

from typing import Any

from dafc.evaluation.audit_transitionguard_postcheck_baselines import DETECTORS


def markdown_report(report: dict[str, Any]) -> str:
    natural = report["natural"]["pooled"]
    boot = report["natural"]["advanced_contract_minus_absolute_bootstrap"]["pooled"]
    adjacent = report["faults"]["adjacent"]["pooled"]
    drop = report["faults"]["drop"]["pooled"]
    gate = report["interpretation_gate"]
    lines = [
        "# TransitionGuard V6 post-check component-baseline audit",
        "",
        f"Decision gate: **{('PASS' if gate['passed'] else 'FAIL')}**.",
        "",
        "This is the preregistered, read-only secondary analysis of frozen V5",
        "logs. It is not a new policy run or a published-system reproduction.",
        "",
        "## Natural observed guarded actions",
        "",
        "| Detector | Accepted advanced | Advanced rate | Accepted adverse | Adverse rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for detector in DETECTORS:
        row = natural["detectors"][detector]
        lines.append(
            f"| {detector} | {row['accepted_advanced']} | {100 * row['accepted_advanced_rate']:.1f}% | {row['accepted_adverse']} | {100 * row['accepted_adverse_rate']:.1f}% |"
        )
    relative_only = natural["relative_only_acceptances"]
    lines.extend(
        [
            "",
            f"The complete contract rescued {relative_only['count']} decisions rejected by the absolute template: {relative_only['semantics']}.",
            f"Its paired advanced-action gain was {100 * boot['estimate']:.1f} points (source-cluster bootstrap 95% CI [{100 * boot['confidence_interval_95'][0]:.1f}, {100 * boot['confidence_interval_95'][1]:.1f}]).",
            "",
            "## Matched fault steps",
            "",
            "| Detector | Drop rejected | Adjacent rejected |",
            "|---|---:|---:|",
        ]
    )
    for detector in DETECTORS:
        drop_row = drop["detectors"][detector]
        adjacent_row = adjacent["detectors"][detector]
        lines.append(
            f"| {detector} | {drop_row['rejected']}/{drop['evaluated_steps']} | {adjacent_row['rejected']}/{adjacent['evaluated_steps']} |"
        )
    lines.extend(["", "## Predeclared gate", ""])
    for name, passed in gate["checks"].items():
        lines.append(f"- [{('x' if passed else ' ')}] `{name}`")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "A pass supports only a component-level coverage--fault-rejection",
            "tradeoff over the logged change-only and absolute-template checks.",
            "It does not show natural episode-success gains, safety, unseen-instance",
            "transfer, or superiority to PreAct, GPA, AutoRPA, or ViBR.",
            "",
        ]
    )
    return "\n".join(lines)


def latex_numbers(report: dict[str, Any]) -> str:
    """Render mechanically bound macros for prose outside the result table."""
    natural = report["natural"]["pooled"]
    bootstrap = report["natural"]["advanced_contract_minus_absolute_bootstrap"]["pooled"]
    drop = report["faults"]["drop"]["pooled"]
    adjacent = report["faults"]["adjacent"]["pooled"]
    relative_only = natural["relative_only_acceptances"]
    semantic_totals = natural["semantic_totals"]
    contract = natural["detectors"]["transition_contract"]
    absolute = natural["detectors"]["absolute_template"]
    values = {
        "TGPostNaturalActions": natural["evaluated_steps"],
        "TGPostAdvancedActions": semantic_totals["advanced"],
        "TGPostAdverseActions": semantic_totals.get("nonadvancing", 0) + semantic_totals.get("harmful", 0),
        "TGPostContractAdvancedRate": f"{100 * contract['accepted_advanced_rate']:.1f}\\%",
        "TGPostAbsoluteAdvancedRate": f"{100 * absolute['accepted_advanced_rate']:.1f}\\%",
        "TGPostAdvancedGain": f"{100 * bootstrap['estimate']:.1f}",
        "TGPostAdvancedGainCILow": f"{100 * bootstrap['confidence_interval_95'][0]:.1f}",
        "TGPostAdvancedGainCIHigh": f"{100 * bootstrap['confidence_interval_95'][1]:.1f}",
        "TGPostRelativeOnly": relative_only["count"],
        "TGPostRelativeOnlyAdvanced": relative_only["semantics"]["advanced"],
        "TGPostRelativeOnlyNonadvancing": relative_only["semantics"]["nonadvancing"],
        "TGPostRelativeOnlyHarmful": relative_only["semantics"]["harmful"],
        "TGPostDropRejected": drop["detectors"]["transition_contract"]["rejected"],
        "TGPostAdjacentRejected": adjacent["detectors"]["transition_contract"]["rejected"],
        "TGPostChangeAdjacentRejected": adjacent["detectors"]["change_only"]["rejected"],
        "TGPostAbsoluteAdjacentRejected": adjacent["detectors"]["absolute_template"]["rejected"],
    }
    return "\n".join((f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items())) + "\n"


def latex_table(report: dict[str, Any]) -> str:
    """Render the compact, claim-bounded table used by the manuscript."""
    natural = report["natural"]["pooled"]
    adjacent = report["faults"]["adjacent"]["pooled"]
    drop = report["faults"]["drop"]["pooled"]
    labels = {
        "change_only": "Change only",
        "absolute_template": "Absolute template",
        "directional_effect": "Directional effect",
        "transition_contract": "\\textbf{Complete contract}",
    }
    lines = [
        "% Generated by scripts/audit_transitionguard_postcheck_baselines.py.",
        "% Secondary logged-component analysis; not a new policy run.",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrr@{}}",
        "\\toprule",
        "Post-check & Adv. acc. & Adverse acc. & Drop rej. & Adj. rej. \\\\",
        "\\midrule",
    ]
    for detector in DETECTORS:
        natural_row = natural["detectors"][detector]
        drop_row = drop["detectors"][detector]
        adjacent_row = adjacent["detectors"][detector]
        lines.append(
            f"{labels[detector]} & {100 * natural_row['accepted_advanced_rate']:.1f} & {100 * natural_row['accepted_adverse_rate']:.1f} & {drop_row['rejected']}/{drop['evaluated_steps']} & {adjacent_row['rejected']}/{adjacent['evaluated_steps']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\caption{Same-observation post-check baselines. Natural columns are acceptance percentages over 1,087 independently advancing and 160 harmful-or-nonadvancing guarded actions; fault columns are rejected injections. This preregistered read-only secondary analysis changes no V5 decision or threshold and implies no counterfactual episode success.}",
            "\\label{tab:tg-postcheck-baselines}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)
