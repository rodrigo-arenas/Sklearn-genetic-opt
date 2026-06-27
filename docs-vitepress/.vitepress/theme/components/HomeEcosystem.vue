<template>
  <section class="home-section home-ecosystem">
    <div class="home-container">
      <h2 class="section-title">Works With Your Stack</h2>
      <p class="section-subtitle">
        Any scikit-learn compatible estimator works — including the most popular gradient boosting libraries.
      </p>

      <div class="ecosystem-groups">
        <div v-for="group in groups" :key="group.label" class="eco-group">
          <span class="eco-label">{{ group.label }}</span>
          <div class="eco-chips">
            <a
              v-for="lib in group.libs"
              :key="lib.name"
              :href="lib.link"
              class="eco-chip"
              :title="lib.desc"
            >
              {{ lib.name }}
            </a>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const groups = [
  {
    label: 'Core',
    libs: [
      { name: 'scikit-learn', link: 'https://scikit-learn.org', desc: 'All sklearn estimators' },
      { name: 'numpy', link: 'https://numpy.org', desc: 'Numerical arrays' },
      { name: 'DEAP', link: 'https://deap.readthedocs.io', desc: 'Evolutionary algorithms' },
    ],
  },
  {
    label: 'Gradient Boosting',
    libs: [
      { name: 'XGBoost', link: 'https://xgboost.readthedocs.io', desc: 'XGBClassifier / XGBRegressor' },
      { name: 'LightGBM', link: 'https://lightgbm.readthedocs.io', desc: 'LGBMClassifier / LGBMRegressor' },
      { name: 'CatBoost', link: 'https://catboost.ai', desc: 'CatBoostClassifier / CatBoostRegressor' },
      { name: 'HistGBM', link: 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html', desc: 'sklearn HistGradientBoosting' },
    ],
  },
  {
    label: 'Experiment Tracking',
    libs: [
      { name: 'MLflow', link: 'https://mlflow.org', desc: 'MLflowConfig built-in' },
      { name: 'TensorBoard', link: 'https://www.tensorflow.org/tensorboard', desc: 'TensorBoard callback' },
    ],
  },
  {
    label: 'Parallelism',
    libs: [
      { name: 'joblib', link: 'https://joblib.readthedocs.io', desc: 'n_jobs=-1 parallel evaluation' },
      { name: 'loky', link: 'https://loky.readthedocs.io', desc: 'loky backend for workers' },
    ],
  },
  {
    label: 'Visualization',
    libs: [
      { name: 'matplotlib', link: 'https://matplotlib.org', desc: 'All plotting functions' },
      { name: 'seaborn', link: 'https://seaborn.pydata.org', desc: 'Optional extra' },
    ],
  },
]
</script>

<style scoped>
.home-ecosystem {
  background: var(--vp-c-bg);
}

.ecosystem-groups {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.eco-group {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  flex-wrap: wrap;
}

.eco-label {
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--vp-c-text-3);
  min-width: 130px;
  padding-top: 0.4rem;
}

.eco-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.eco-chip {
  display: inline-block;
  padding: 0.3rem 0.8rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--vp-c-text-1);
  text-decoration: none;
  background: var(--vp-c-bg-soft);
  transition: border-color 0.15s, color 0.15s;
}

.eco-chip:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

@media (max-width: 640px) {
  .eco-label {
    min-width: 100%;
    padding-top: 0;
  }
}
</style>
