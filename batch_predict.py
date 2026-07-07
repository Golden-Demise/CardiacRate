import sys
import subprocess
# python acdc_random_forest.py predict --model D:\CardiacRate\dataset_acdc\classification\rf_v1\acdc_random_forest.joblib --facts_path D:\CardiacRate\dataset_acdc\facts_gt\testing\patient101_facts.json --out_json D:\CardiacRate\dataset_acdc\classification\patient101_rf_prediction.json
for i in range(101,151):
    f_p = "D://CardiacRate//dataset_acdc//facts_gt//testing//patient"+str(i)+"_facts.json"
    o_p = "D://CardiacRate//dataset_acdc//classification//patient"+str(i)+"_rf_prediction.json"
    cmd = [
        sys.executable,
        "D://CardiacRate//acdc_random_forest.py",
        "predict",
        "--model",
        "D://CardiacRate//dataset_acdc//classification//rf_v1//acdc_random_forest.joblib",
        "--facts_path",
        f_p,
        "--out_json",
        o_p
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"make_facts_acdc.py failed for {i}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )