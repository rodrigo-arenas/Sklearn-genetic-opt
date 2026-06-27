<template>
  <section class="home-section home-learning-paths">
    <div class="home-container">
      <h2 class="section-title">Learning Paths</h2>
      <p class="section-subtitle">
        Choose based on your experience. Each path is a curated sequence of docs.
      </p>

      <div class="paths-grid">
        <div v-for="path in paths" :key="path.title" class="path-card">
          <div class="path-header">
            <span class="path-icon">{{ path.icon }}</span>
            <span :class="['path-level', `level-${path.level}`]">{{ path.level }}</span>
          </div>
          <h3 class="path-title">{{ path.title }}</h3>
          <p class="path-desc">{{ path.desc }}</p>
          <ol class="path-steps">
            <li v-for="step in path.steps" :key="step.text">
              <a :href="step.link">{{ step.text }}</a>
            </li>
          </ol>
          <a :href="path.cta.link" class="path-cta">{{ path.cta.text }} →</a>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const paths = [
  {
    icon: '🌱',
    level: 'Beginner',
    title: 'First Optimization',
    desc: 'Never used a genetic algorithm? Start here. You\'ll run your first GASearchCV in under 10 minutes.',
    steps: [
      { text: 'Installation', link: '/stable/' },
      { text: 'How hyperparameter optimization works', link: '/versions/latest/guide/how-hyperparameter-optimization-works' },
      { text: 'Tune a Random Forest (tutorial)', link: '/versions/latest/tutorials/tune-random-forest' },
      { text: 'Basic usage example', link: '/versions/latest/examples/basic-usage' },
    ],
    cta: { text: 'Start the beginner path', link: '/stable/' },
  },
  {
    icon: '⚙️',
    level: 'Intermediate',
    title: 'Production Tuning',
    desc: 'You\'ve run a basic search. Now learn callbacks, parallel evaluation, MLflow logging, and pipelines.',
    steps: [
      { text: 'Choosing the right search space', link: '/versions/latest/guide/choosing-search-spaces' },
      { text: 'Common tuning mistakes to avoid', link: '/versions/latest/guide/common-mistakes' },
      { text: 'Tune XGBoost / LightGBM', link: '/versions/latest/tutorials/tune-xgboost' },
      { text: 'Callbacks & early stopping', link: '/versions/latest/tutorials/callbacks' },
    ],
    cta: { text: 'Browse intermediate recipes', link: '/versions/latest/recipes/' },
  },
  {
    icon: '🏆',
    level: 'Advanced',
    title: 'Evolutionary Mastery',
    desc: 'Deep-dive into GA mechanics, feature selection, custom operators, and multi-metric optimization.',
    steps: [
      { text: 'Feature selection with GAFeatureSelectionCV', link: '/versions/latest/tutorials/feature-selection-with-genetic-algorithms' },
      { text: 'Warm-start and custom initialization', link: '/versions/latest/recipes/advanced/warm-start' },
      { text: 'Benchmarks vs Optuna & random search', link: '/versions/latest/benchmarks' },
      { text: 'API reference — all parameters', link: '/versions/latest/api/GASearchCV' },
    ],
    cta: { text: 'View advanced recipes', link: '/versions/latest/recipes/advanced/' },
  },
]
</script>

<style scoped>
.home-learning-paths {
  background: var(--vp-c-bg-soft);
}

.paths-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
}

.path-card {
  display: flex;
  flex-direction: column;
  padding: 1.5rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
}

.path-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}

.path-icon {
  font-size: 1.75rem;
}

.path-level {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
}

.level-Beginner     { background: #dcfce7; color: #166534; }
.level-Intermediate { background: #fef9c3; color: #854d0e; }
.level-Advanced     { background: #ede9fe; color: #5b21b6; }

.dark .level-Beginner     { background: #14532d; color: #86efac; }
.dark .level-Intermediate { background: #422006; color: #fde68a; }
.dark .level-Advanced     { background: #2e1065; color: #c4b5fd; }

.path-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin: 0 0 0.5rem;
}

.path-desc {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  line-height: 1.55;
  margin: 0 0 1rem;
}

.path-steps {
  padding-left: 1.25rem;
  margin: 0 0 1.25rem;
  flex-grow: 1;
}

.path-steps li {
  font-size: 0.85rem;
  margin-bottom: 0.4rem;
}

.path-steps a {
  color: var(--vp-c-brand-1);
  text-decoration: none;
}

.path-steps a:hover {
  text-decoration: underline;
}

.path-cta {
  display: inline-block;
  margin-top: auto;
  padding: 0.55rem 1rem;
  border: 1px solid var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  border-radius: 6px;
  font-size: 0.85rem;
  font-weight: 600;
  text-decoration: none;
  text-align: center;
  transition: background 0.15s, color 0.15s;
}

.path-cta:hover {
  background: var(--vp-c-brand-1);
  color: white;
}
</style>
