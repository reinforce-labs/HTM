`datasets.yaml` is the central configuration file for the data generation pipeline. It specifies every dataset, its role (`train`, `val`, or `test`), target conversation count, optional minimum conversation length, and which labeling stages are applied.

For example:

```yaml
xguard:
  role: train
  count: 15000
  min_turns: 7
  label: teacher+step
```

## Fields

- **`role`** – Dataset split used throughout the pipeline (`train`, `val`, or `test`).
- **`count`** – Maximum number of conversations to include after filtering.
- **`min_turns`** *(optional)* – Minimum number of user–assistant exchanges required before labeling.
- **`label`** – Labeling stages to run (`teacher`, `step`, or `teacher+step`). Test datasets typically use only `step` labels.

Keeping these settings in a single configuration file ensures that dataset splits, filtering, and labeling are defined once and shared consistently across the pipeline.