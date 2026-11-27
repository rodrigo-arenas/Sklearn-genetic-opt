<template>
  <section class="home-section home-visuals" aria-labelledby="home-visuals-title">
    <div class="home-container">
      <h2 id="home-visuals-title" class="section-title">Inside a genetic search</h2>
      <p class="section-subtitle">
        Explore one random forest search on scikit-learn’s breast cancer dataset.
        Open a chart for the full figure and Python example.
      </p>

      <div class="visuals-grid">
        <a
          v-for="item in visuals"
          :key="item.fn"
          :href="gallery + '#' + item.anchor"
          class="visual-card"
        >
          <h3 class="visual-title">{{ item.title }}</h3>
          <img
            :src="'/images/home-gallery/' + item.image + '.svg'"
            :alt="item.alt"
            class="visual-img"
            width="324"
            height="180"
            loading="lazy"
            decoding="async"
          />
          <span class="visual-caption">
            <code>{{ item.fn }}()</code><span aria-hidden="true">→</span>
          </span>
        </a>
      </div>

      <div class="visuals-cta">
        <a :href="gallery" class="visuals-link">View the full plotting gallery →</a>
      </div>
    </div>
  </section>
</template>

<script setup>
const gallery = '/stable/examples/plotting-gallery'
const visuals = [
  {
    image: 'fitness',
    title: 'Fitness by generation',
    alt: 'Best observed ROC AUC rises and levels off. The population mean fluctuates below it.',
    fn: 'plot_fitness_evolution',
    anchor: 'fitness-evolution',
  },
  {
    image: 'overview',
    title: 'Search overview',
    alt: 'Best cross-validation ROC AUC and total evaluations, with duplicate replacements by generation.',
    fn: 'plot_search_overview',
    anchor: 'overview-dashboard',
  },
  {
    image: 'correlation',
    title: 'Parameter correlations',
    alt: 'Pearson correlations between the sampled tree count, depth, minimum split size and feature fraction.',
    fn: 'plot_search_space',
    anchor: 'search-space-exploration',
  },
  {
    image: 'parameters',
    title: 'Tree count by evaluation',
    alt: 'Each point shows the number of trees in one evaluated forest, colored by mean cross-validation ROC AUC.',
    fn: 'plot_parameter_evolution',
    anchor: 'parameter-exploration-over-time',
  },
  {
    image: 'diversity',
    title: 'Population diversity',
    alt: 'The proportion of unique individuals stays above the genotype diversity measure throughout the run.',
    fn: 'plot_diversity',
    anchor: 'focused-convergence-and-diversity',
  },
  {
    image: 'landscape',
    title: 'Score landscape',
    alt: 'Sampled tree depths and minimum split sizes, colored by mean cross-validation ROC AUC.',
    fn: 'plot_score_landscape',
    anchor: 'score-landscapes',
  },
]
</script>

<style scoped>
.home-visuals {
  background: var(--vp-c-bg-soft);
}

.visuals-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1.25rem;
  margin-bottom: 2rem;
}

.visual-card {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
}

.visual-title {
  margin: 0;
  padding: 1rem 1.2rem 0.35rem;
  font-size: 1.15rem;
  font-weight: 650;
  line-height: 1.35;
  letter-spacing: -0.02em;
}

.visual-img {
  width: 100%;
  height: auto;
  display: block;
  margin-block: auto 0;
}

.visual-caption {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.55rem 1.2rem;
  border-top: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-alt);
  color: var(--vp-c-brand-1);
}

.visual-caption code {
  font-family: var(--vp-font-family-mono);
  font-size: 0.8rem;
}

.visual-card:is(:hover, :focus-visible) {
  border-color: var(--vp-c-brand-1);
}

.visual-card:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 3px;
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

@media (max-width: 1023px) {
  .visuals-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 679px) {
  .visuals-grid { grid-template-columns: minmax(0, 1fr); }
}
</style>
