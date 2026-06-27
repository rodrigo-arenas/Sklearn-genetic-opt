---
title: Scoring Metrics Recipes
description: Copy-paste recipes for optimizing GASearchCV with F1, ROC-AUC, balanced accuracy, MAE, and RMSE.
---

:::warning Development version
You are reading the **latest (dev)** docs. For the stable version, see [stable](/stable/).
:::

# Scoring Metrics Recipes

Five copy-paste recipes for common scoring objectives.

| Recipe | Metric | Task |
|--------|--------|------|
| [Optimize for F1 (binary)](./f1-binary) | `f1` | Classification |
| [Optimize for ROC-AUC](./roc-auc) | `roc_auc` | Classification |
| [Optimize for Balanced Accuracy](./balanced-accuracy) | `balanced_accuracy` | Imbalanced classification |
| [Optimize for MAE](./mae) | `neg_mean_absolute_error` | Regression |
| [Optimize for RMSE](./rmse) | `neg_root_mean_squared_error` | Regression |

## See Also

- [All Recipes](../) — full recipe index
- [Common Tuning Mistakes](../../guide/common-mistakes) — avoiding metric leakage
