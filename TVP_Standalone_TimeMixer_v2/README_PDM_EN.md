# TimeMixer PDM Audit v1

A read-only audit for the existing standalone TVP/TimeMixer project. Drop these files directly into the project root, then run `bash 19_run_pdm_audit.sh`.

It verifies the pinned official code, computes scale-aware decomposition indicators on the **training input only**, analyzes learned PDM activations using an **existing trained checkpoint**, and performs **frozen validation-only module interventions**.

Intervention results are NOT retrained ablation experiments. All model weights remain unchanged. No test labels are accessed.

See `README_PDM_CN.md` for default folder paths, output schema and interpretation.
