import os.path
import subprocess
import re

from runners import RUNNERS_DIRECTORY_PATH


EXPECTED_RESULTS = {
    "AUC": 0.5685,
    "C-for-Benefit (test)": 0.517,
    "C-for-Benefit (train)": 0.503,
    "Retention gain": -0.008,
    "ROI vs historical": -14.4,
    "ATE": 0.03061,
    "Expected uplift": 0.18757,
}

TOLERANCE = 0.002  # ±0.2% relative tolerance


def extract_value(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def test_no_performance_change():
    env = os.environ.copy()
    env.update({
        "TEST_MODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    })

    project_root_dir_path = os.path.dirname(RUNNERS_DIRECTORY_PATH)
    script_path = os.path.join(RUNNERS_DIRECTORY_PATH, 'run_training_and_evaluation.py')

    result = subprocess.run(
        ["python", script_path],
        capture_output=True,
        text=True,
        check=True,
        cwd=project_root_dir_path,
        env=env,
    )

    assert result.returncode == 0, (
        "run_training_and_evaluation.py failed\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )
    output = result.stdout

    actual = {
        "AUC": extract_value(r"AUC:\s*([0-9.]+)", output),
        "C-for-Benefit (test)": extract_value(r"C-for-Benefit \(test\):\s*([0-9.]+)", output),
        "C-for-Benefit (train)": extract_value(r"C-for-Benefit \(train\):\s*([0-9.]+)", output),
        "Retention gain": extract_value(r"Retention gain =\s*([-0-9.]+)", output),
        "ROI vs historical": extract_value(r"ROI vs historical:\s*([-0-9.]+)", output),
        "ATE": extract_value(r"Average Treatment Effect.*:\s*([0-9.]+)", output),
        "Expected uplift": extract_value(r"Expected uplift.*:\s*([0-9.]+)", output),
    }

    for key, expected in EXPECTED_RESULTS.items():
        value = actual.get(key)
        assert value is not None, f"{key} not found in output"
        diff = abs(value - expected)
        assert diff <= TOLERANCE * abs(expected), (
            f"{key} changed! expected={expected:.4f}, got={value:.4f}, diff={diff:.4f}"
        )

    print("Performance check passed. No significant changes detected.")


if __name__ == "__main__":
    test_no_performance_change()
