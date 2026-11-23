# Examples

Sample datasets and Ludwig configs to try dflow. Each folder has `data.csv`, `ludwig_config.json` and a `README.md` with the curl command.

| Folder | Task | Rows | Split |
| --- | --- | --- | --- |
| `predictive_maintenance_rpm/` | binary (small smoke test) | 60 | default |
| `binary_failure/` | binary classification | 1000 | default |
| `regression_power_output/` | regression | 1000 | default |
| `multiclass_fault_type/` | multiclass | 1000 | explicit 0.8/0.1/0.1 |

Regenerate the synthetic datasets (deterministic, seed 42) with `python examples/generate_examples.py`.
