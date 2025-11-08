import os
import sys
import subprocess
import re

from runners import RUNNERS_DIRECTORY_PATH


EXPECTED_RESULTS = {
    "AUC": 0.616225,
    "C-for-Benefit (test)": 0.512,
    "C-for-Benefit (train)": 0.518,
    "ROI vs historical": -14.4,
    "ATE": 3.263,
    "Expected uplift": 20.741,
}

TOLERANCE = 0.002  # ±0.2% relative tolerance
ABS_FLOOR = 1e-6   # minimum absolute tolerance to handle expected≈0


def extract_value(pattern, text):
    num = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    match = re.search(pattern.replace("([0-9.]+)", num), text)
    return float(match.group(1)) if match else None


def test_no_performance_change():
    env = os.environ.copy()
    env.update({
        "TEST_MODE": "1",
    })

    project_root_dir_path = os.path.dirname(RUNNERS_DIRECTORY_PATH)
    env["PYTHONPATH"] = project_root_dir_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "runners.run_training_and_evaluation",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=project_root_dir_path,
        env=env,
    )

    if result.returncode != 0:
        raise AssertionError(
            "run_training_and_evaluation failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"CWD: {project_root_dir_path}\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    output = result.stdout

    actual = {
        "AUC": extract_value(r"AUC:\s*([0-9.]+)", output),
        "C-for-Benefit (test)": extract_value(r"C-for-Benefit \(test\):\s*([0-9.]+)", output),
        "C-for-Benefit (train)": extract_value(r"C-for-Benefit \(train\):\s*([0-9.]+)", output),
        "ROI vs historical": extract_value(r"ROI vs historical:\s*([-0-9.eE+]+)", output),
        "ATE": extract_value(r"Average Treatment Effect.*:\s*([-0-9.eE+]+)", output),
        "Expected uplift": extract_value(r"Expected uplift.*:\s*([-0-9.eE+]+)", output),
    }

    missing = [k for k, v in actual.items() if v is None]
    assert not missing, f"Metrics not found in output: {missing}\n\nOutput:\n{output}"

    for key, expected in EXPECTED_RESULTS.items():
        value = actual[key]
        tol = max(TOLERANCE * max(abs(expected), 1.0), ABS_FLOOR)
        diff = abs(value - expected)
        assert diff <= tol, (
            f"{key} changed! expected={expected:.6f}, got={value:.6f}, "
            f"diff={diff:.6f}, tol={tol:.6f}"
        )

    print("Performance check passed. No significant changes detected.")
    for k, v in actual.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    test_no_performance_change()
