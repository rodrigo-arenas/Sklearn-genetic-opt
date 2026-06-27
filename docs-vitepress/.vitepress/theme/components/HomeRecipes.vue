<template>
  <section class="home-section home-recipes">
    <div class="home-container">
      <h2 class="section-title">Solve Real Problems</h2>
      <p class="section-subtitle">
        Copy-paste recipes for common tasks. Each one runs as-is and takes 5–10 minutes to read.
      </p>

      <div class="recipe-grid">
        <a
          v-for="recipe in recipes"
          :key="recipe.title"
          :href="recipe.link"
          class="recipe-card"
        >
          <div class="recipe-top">
            <span class="recipe-icon">{{ recipe.icon }}</span>
            <span class="recipe-category">{{ recipe.category }}</span>
          </div>
          <h3 class="recipe-title">{{ recipe.title }}</h3>
          <p class="recipe-desc">{{ recipe.desc }}</p>
          <div class="recipe-meta">
            <span class="recipe-time">⏱ {{ recipe.time }}</span>
            <span :class="['recipe-difficulty', `diff-${recipe.difficulty.toLowerCase()}`]">
              {{ recipe.difficulty }}
            </span>
          </div>
        </a>
      </div>

      <div class="recipes-cta">
        <a href="/versions/latest/recipes/" class="browse-all-btn">Browse all 30 recipes →</a>
      </div>
    </div>
  </section>
</template>

<script setup>
const recipes = [
  {
    icon: '🌲',
    category: 'Classification',
    title: 'Tune RandomForestClassifier',
    desc: '7-parameter joint search. Which parameters matter, classification and regression variants.',
    time: '8 min',
    difficulty: 'Beginner',
    link: '/versions/latest/recipes/classification/random-forest-classifier',
  },
  {
    icon: '⚡',
    category: 'Classification',
    title: 'Tune XGBClassifier',
    desc: 'learning_rate × n_estimators interaction, subsample, colsample_bytree — 8-parameter search.',
    time: '10 min',
    difficulty: 'Intermediate',
    link: '/versions/latest/recipes/classification/xgboost-classifier',
  },
  {
    icon: '🔦',
    category: 'Classification',
    title: 'Tune LGBMClassifier',
    desc: 'num_leaves vs max_depth, min_child_samples, feature_fraction — LightGBM-specific parameters.',
    time: '10 min',
    difficulty: 'Intermediate',
    link: '/versions/latest/recipes/classification/lightgbm-classifier',
  },
  {
    icon: '🎯',
    category: 'Feature Selection',
    title: 'High-Dimensional Feature Selection',
    desc: 'GAFeatureSelectionCV on datasets with 100+ features. Faster than RFE, respects interactions.',
    time: '7 min',
    difficulty: 'Intermediate',
    link: '/versions/latest/recipes/feature-selection/high-dimensional',
  },
  {
    icon: '🔗',
    category: 'Pipelines',
    title: 'Preprocessing + Estimator Pipeline',
    desc: 'Tune preprocessing parameters and model hyperparameters jointly in a single Pipeline.',
    time: '8 min',
    difficulty: 'Intermediate',
    link: '/versions/latest/recipes/pipelines/preprocessing-pipeline',
  },
  {
    icon: '📋',
    category: 'Scoring Metrics',
    title: 'Optimize for ROC-AUC',
    desc: 'Set scoring="roc_auc", handle predict_proba requirements, multi-class extension.',
    time: '5 min',
    difficulty: 'Beginner',
    link: '/versions/latest/recipes/metrics/roc-auc',
  },
  {
    icon: '🧪',
    category: 'Advanced',
    title: 'Warm-Start with Known-Good Configs',
    desc: 'Seed the initial population with hyperparameters from a previous run or domain knowledge.',
    time: '6 min',
    difficulty: 'Advanced',
    link: '/versions/latest/recipes/advanced/warm-start',
  },
  {
    icon: '🔬',
    category: 'Regression',
    title: 'Tune XGBRegressor',
    desc: 'Same parameters as XGBClassifier but optimized for RMSE, MAE, and R² objectives.',
    time: '8 min',
    difficulty: 'Intermediate',
    link: '/versions/latest/recipes/regression/xgboost-regressor',
  },
]
</script>

<style scoped>
.home-recipes {
  background: var(--vp-c-bg);
}

.recipe-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.25rem;
  margin-bottom: 2rem;
}

.recipe-card {
  display: flex;
  flex-direction: column;
  padding: 1.25rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  text-decoration: none;
  transition: border-color 0.2s, transform 0.15s;
}

.recipe-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
}

.recipe-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}

.recipe-icon {
  font-size: 1.5rem;
}

.recipe-category {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
}

.recipe-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin: 0 0 0.5rem;
  line-height: 1.4;
}

.recipe-desc {
  font-size: 0.82rem;
  color: var(--vp-c-text-2);
  line-height: 1.55;
  margin: 0 0 auto;
  flex-grow: 1;
}

.recipe-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 1rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--vp-c-divider);
}

.recipe-time {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
}

.recipe-difficulty {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
}

.diff-beginner    { background: #dcfce7; color: #166534; }
.diff-intermediate { background: #fef9c3; color: #854d0e; }
.diff-advanced    { background: #fee2e2; color: #991b1b; }

.dark .diff-beginner    { background: #14532d; color: #86efac; }
.dark .diff-intermediate { background: #422006; color: #fde68a; }
.dark .diff-advanced    { background: #450a0a; color: #fca5a5; }

.recipes-cta {
  text-align: center;
}

.browse-all-btn {
  display: inline-block;
  padding: 0.7rem 1.75rem;
  background: var(--vp-c-brand-1);
  color: white;
  border-radius: 6px;
  font-weight: 600;
  font-size: 0.9rem;
  text-decoration: none;
  transition: opacity 0.15s;
}

.browse-all-btn:hover {
  opacity: 0.85;
}
</style>
