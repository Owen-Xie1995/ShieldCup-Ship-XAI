#!/usr/bin/env python3
"""Compare reviewed admission/scorer AST using synthetic tables; not a sandbox."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import sys
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path: Path, function_name: str, constants=()):
    # A selected function can execute arbitrary code when called. This helper is
    # for source that a reviewer has already read, never untrusted submissions.
    import numpy as np
    import pandas as pd

    tree = ast.parse(path.read_bytes(), filename=path.name)
    names, nodes, found = set(constants), [], set()
    for node in tree.body:
        selected_names = set()
        if isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & names:
                if not targets <= names or len(node.targets) != len(targets):
                    raise ValueError("Selected constants must use simple assignments")
                allowed = (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.Constant,
                           ast.UnaryOp, ast.UAdd, ast.USub, ast.Attribute, ast.Name, ast.Load)
                for item in ast.walk(node.value):
                    if not isinstance(item, allowed):
                        raise ValueError("Constants must be literals or math.inf")
                    if isinstance(item, ast.Name) and item.id != "math":
                        raise ValueError("Only math.inf is supported as a named constant")
                    if isinstance(item, ast.Attribute) and not (isinstance(item.value, ast.Name) and item.value.id == "math" and item.attr == "inf"):
                        raise ValueError("Only math.inf is supported as an attribute constant")
                selected_names = targets
        elif isinstance(node, ast.FunctionDef) and node.name == function_name:
            if node.decorator_list or node.args.defaults or any(value is not None for value in node.args.kw_defaults):
                raise ValueError("Selected function must not have decorators or evaluated defaults")
            selected_names = {node.name}
        if selected_names:
            if found & selected_names:
                raise ValueError("Duplicate selected definition")
            nodes.append(node)
            found.update(selected_names)
    if found != names | {function_name}:
        raise ValueError(f"Missing selected definitions: {sorted((names | {function_name}) - found)}")
    selected = ast.Module(body=nodes, type_ignores=[])
    namespace = {"np": np, "pd": pd, "math": math}
    exec(compile(selected, path.name + " [reviewed AST only]", "exec"), namespace)
    return namespace[function_name], nodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-runner", type=Path, required=True)
    parser.add_argument("--v2-runner", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-source", action="store_true", help="Confirm the selected functions were manually reviewed before executing them")
    args = parser.parse_args()
    if not args.reviewed_source:
        parser.error("Review strict_admission, score, EXPECTED_COLUMNS and METRIC_RANGES first; then pass --reviewed-source. AST selection is not a security sandbox.")
    sources = (args.v1_runner.resolve(), args.v2_runner.resolve(), args.scorer.resolve())
    if not all(path.is_file() for path in sources):
        parser.error("All three source paths must be existing files")
    if args.output.resolve() in sources:
        parser.error("Output must not overwrite source")
    import numpy as np
    import pandas as pd

    constants = ("EXPECTED_COLUMNS", "METRIC_RANGES")
    admission_v1, nodes_v1 = extract(sources[0], "strict_admission", constants)
    admission_v2, nodes_v2 = extract(sources[1], "strict_admission", constants)
    score, score_nodes = extract(sources[2], "score")
    function_v1 = next(node for node in nodes_v1 if isinstance(node, ast.FunctionDef))
    function_v2 = next(node for node in nodes_v2 if isinstance(node, ast.FunctionDef))
    if ast.dump(function_v1) != ast.dump(function_v2):
        raise ValueError("Admission function changed beyond constants; review test expectations before comparing")

    columns = ["Id", "Faithfulness", "Max_Sensitivity", "Sparsity"]


    def frame(**changes):
        values = {"Id": 1, "Faithfulness": 0.5, "Max_Sensitivity": 0.8, "Sparsity": 0.9}
        values.update(changes)
        return pd.DataFrame([values], columns=columns)


    cases = []


    def case(name, data, v2_expected, v1_expected=None, expected_rows=1):
        cases.append((name, data, v2_expected, v2_expected if v1_expected is None else v1_expected, expected_rows))


    case("baseline", frame(), True)
    case("sensitivity_zero", frame(Max_Sensitivity=0.0), True)
    case("sensitivity_exactly_two", frame(Max_Sensitivity=2.0), True)
    case("sensitivity_just_above_two", frame(Max_Sensitivity=2.000001), True, False)
    case("sensitivity_three", frame(Max_Sensitivity=3.0), True, False)
    case("sensitivity_large_finite", frame(Max_Sensitivity=1e300), True, False)
    case("sensitivity_negative", frame(Max_Sensitivity=-0.001), False)
    case("sensitivity_nan", frame(Max_Sensitivity=np.nan), False)
    case("sensitivity_positive_infinity", frame(Max_Sensitivity=np.inf), False)
    case("sensitivity_negative_infinity", frame(Max_Sensitivity=-np.inf), False)
    case("sensitivity_non_numeric", frame(Max_Sensitivity="invalid"), False)
    case("faithfulness_negative_valid", frame(Faithfulness=-0.5), True)
    case("faithfulness_lower_endpoint", frame(Faithfulness=-1.0), True)
    case("faithfulness_upper_endpoint", frame(Faithfulness=1.0), True)
    case("faithfulness_below_lower", frame(Faithfulness=-1.001), False)
    case("faithfulness_above_upper", frame(Faithfulness=1.001), False)
    case("faithfulness_nan", frame(Faithfulness=np.nan), False)
    case("faithfulness_infinity", frame(Faithfulness=np.inf), False)
    case("sparsity_lower_endpoint", frame(Sparsity=0.0), True)
    case("sparsity_upper_endpoint", frame(Sparsity=1.0), True)
    case("sparsity_negative", frame(Sparsity=-0.001), False)
    case("sparsity_above_upper", frame(Sparsity=1.001), False)
    case("sparsity_nan", frame(Sparsity=np.nan), False)
    case("sparsity_infinity", frame(Sparsity=np.inf), False)
    case("wrong_column_order", frame()[["Id", "Max_Sensitivity", "Faithfulness", "Sparsity"]], False)
    case("missing_column", frame().drop(columns=["Sparsity"]), False)
    case("extra_column", frame().assign(extra=1), False)
    case("wrong_row_count", frame(), False, expected_rows=2)
    case("id_wrong_start", frame(Id=2), False)
    case("id_fractional", frame(Id=1.5), False)
    case("id_nan", frame(Id=np.nan), False)
    case("id_non_numeric", frame(Id="invalid"), False)
    duplicate = pd.concat([frame(), frame()], ignore_index=True)
    case("id_duplicate", duplicate, False, expected_rows=2)
    wrong_order = pd.concat([frame(Id=2), frame(Id=1)], ignore_index=True)
    case("id_wrong_order", wrong_order, False, expected_rows=2)

    results = []
    for name, data, expected_v2, expected_v1, expected_rows in cases:
        actual_v1 = admission_v1(data.copy(), expected_rows)
        actual_v2 = admission_v2(data.copy(), expected_rows)
        matched = actual_v1["passed"] == expected_v1 and actual_v2["passed"] == expected_v2
        results.append({
            "name": name, "expected_v1": expected_v1, "expected_v2": expected_v2,
            "actual_v1": actual_v1["passed"], "actual_v2": actual_v2["passed"],
            "v1_failures": actual_v1["failures"], "v2_failures": actual_v2["failures"],
            "test_passed": matched,
        })

    solution = pd.DataFrame({"Id": [1]})
    scorer_results = []
    for sensitivity, expected in [(0.0, 87.0), (1.0, 72.0), (2.0, 57.0), (3.0, 57.0), (1e300, 57.0)]:
        actual = score(solution.copy(), frame(Max_Sensitivity=sensitivity), "Id")
        scorer_results.append({
            "sensitivity": sensitivity, "faithfulness": 0.5, "sparsity": 0.9,
            "expected_score": expected, "actual_score": actual,
            "test_passed": math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
        })

    record = {
        "status": "PASS" if all(result["test_passed"] for result in results + scorer_results) else "FAIL",
        "executed": "Reviewed strict_admission and score AST only, plus EXPECTED_COLUMNS and METRIC_RANGES constants",
        "runner_imported": False, "torch_imported": "torch" in sys.modules, "model_or_weights_loaded": False,
        "full_model_or_jetson_test": False,
        "device": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "sources": {role: digest(path) for role, path in zip(("v1_runner", "v2_runner", "scorer"), sources)},
        "admission_function_ast_unchanged": True,
        "admission_cases": len(results), "admission_passed": sum(result["test_passed"] for result in results),
        "scorer_cases": len(scorer_results), "scorer_passed": sum(result["test_passed"] for result in scorer_results),
        "results": results, "scorer_results": scorer_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: record[key] for key in ["status", "device", "admission_cases", "admission_passed", "scorer_cases", "scorer_passed", "runner_imported", "torch_imported", "model_or_weights_loaded"]}, ensure_ascii=False, indent=2))
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
