<template>
  <section class="home-section home-visuals">
    <div class="home-container">
      <h2 class="section-title">See What You Get</h2>
      <p class="section-subtitle">
        Rich built-in visualizations — from fitness evolution to parameter interaction heatmaps.
      </p>

      <div class="visuals-grid">
        <div v-for="item in visuals" :key="item.title" class="visual-card">
          <div class="visual-img-wrap">
            <img
              :src="item.src"
              :alt="item.alt"
              class="visual-img"
              loading="lazy"
            />
          </div>
          <div class="visual-caption">
            <span class="visual-fn">{{ item.fn }}</span>
            <p class="visual-title">{{ item.title }}</p>
          </div>
        </div>
      </div>

      <div class="visuals-cta">
        <a href="/versions/latest/examples/plotting-gallery" class="visuals-link">
          View the full plotting gallery →
        </a>
      </div>
    </div>
  </section>
</template>

<script setup>
const visuals = [
  {
    src: '/images/plotting_gallery_fitness_evolution.png',
    alt: 'Fitness evolution over generations',
    fn: 'plot_fitness_evolution()',
    title: 'Track score improvement across generations',
  },
  {
    src: '/images/plotting_gallery_search_overview.png',
    alt: 'Search overview dashboard',
    fn: 'plot_search_overview()',
    title: 'Full-run dashboard: diversity, stagnation, scores',
  },
  {
    src: '/images/tune_xgboost_interaction.png',
    alt: 'Parameter interaction heatmap for XGBoost',
    fn: 'plot_search_space()',
    title: 'Discover learning_rate × n_estimators interactions',
  },
  {
    src: '/images/plotting_gallery_parameter_evolution.png',
    alt: 'Parameter evolution over generations',
    fn: 'plot_parameter_evolution()',
    title: 'Watch each hyperparameter converge over time',
  },
  {
    src: '/images/plotting_gallery_diversity.png',
    alt: 'Population diversity over generations',
    fn: 'plot_diversity()',
    title: 'Monitor genetic diversity and diversity control events',
  },
  {
    src: '/images/plotting_gallery_score_landscape.png',
    alt: 'Score landscape visualization',
    fn: 'plot_score_landscape()',
    title: 'Visualize score surface across parameter pairs',
  },
]
</script>

<style scoped>
.home-visuals {
  background: var(--vp-c-bg-soft);
}

.visuals-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.25rem;
  margin-bottom: 2rem;
}

.visual-card {
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  overflow: hidden;
  background: var(--vp-c-bg);
  transition: border-color 0.2s;
}

.visual-card:hover {
  border-color: var(--vp-c-brand-1);
}

.visual-img-wrap {
  aspect-ratio: 16 / 10;
  overflow: hidden;
  background: #0d0f14;
}

.visual-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.visual-caption {
  padding: 0.75rem 1rem;
}

.visual-fn {
  font-family: var(--vp-font-family-mono);
  font-size: 0.72rem;
  color: var(--vp-c-brand-1);
  display: block;
  margin-bottom: 0.25rem;
}

.visual-title {
  font-size: 0.82rem;
  color: var(--vp-c-text-2);
  margin: 0;
  line-height: 1.4;
}

.visuals-cta {
  text-align: center;
}

.visuals-link {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  text-decoration: none;
}

.visuals-link:hover {
  text-decoration: underline;
}
</style>
