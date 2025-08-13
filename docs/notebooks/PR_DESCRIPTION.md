### Summary
Adds a new Jupyter notebook example to the documentation:  
**`Outlier_Detection_IsolationForest_GASearchCV.ipynb`**

### Motivation
- v0.12.0 added native Outlier Detection Support.
- The docs lacked a dedicated example demonstrating `GASearchCV` with `IsolationForest`.

### What’s Included
- Unsupervised tuning with `GASearchCV(scoring=None)`
- Synthetic dataset with injected outliers
- Visualization using `plot_fitness_evolution`
- Optional evaluation using ROC-AUC

### Related Issue
Addresses: Improve documentation and examples (#101)

### Checklist
- [ ] Notebook runs locally end-to-end
- [ ] Added to docs notebooks folder
- [ ] Updated docs index/toctree if required
