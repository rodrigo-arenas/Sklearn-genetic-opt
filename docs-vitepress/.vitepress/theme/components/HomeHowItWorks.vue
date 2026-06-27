<template>
  <section class="home-section home-how-it-works">
    <div class="home-container">
      <h2 class="section-title">How It Works</h2>
      <p class="section-subtitle">
        Genetic algorithms mimic natural selection to explore hyperparameter space more efficiently than grid or random search.
      </p>

      <div class="steps-grid">
        <div v-for="step in steps" :key="step.number" class="step-card">
          <div class="step-number">{{ step.number }}</div>
          <div class="step-content">
            <h3 class="step-title">{{ step.title }}</h3>
            <p class="step-desc">{{ step.desc }}</p>
          </div>
        </div>
      </div>

      <div class="flow-arrow-row">
        <span v-for="(label, i) in flowLabels" :key="i" class="flow-chip">
          {{ label }}
          <span v-if="i < flowLabels.length - 1" class="flow-arrow">→</span>
        </span>
      </div>
    </div>
  </section>
</template>

<script setup>
const steps = [
  {
    number: '01',
    title: 'Initialize Population',
    desc: 'Latin hypercube sampling generates a diverse initial population covering the search space more evenly than random starts.',
  },
  {
    number: '02',
    title: 'Evaluate Fitness',
    desc: 'Each candidate configuration is cross-validated in parallel. Duplicates are cached — identical configs are never re-evaluated.',
  },
  {
    number: '03',
    title: 'Select & Reproduce',
    desc: 'Tournament selection picks the strongest individuals. Uniform crossover and mutation create offspring with new combinations.',
  },
  {
    number: '04',
    title: 'Converge or Continue',
    desc: 'Diversity control and fitness sharing prevent premature convergence. Callbacks stop the search when it plateaus or hits a budget.',
  },
]

const flowLabels = ['Population', 'Evaluate', 'Select', 'Crossover', 'Mutate', 'Best params']
</script>

<style scoped>
.home-how-it-works {
  background: var(--vp-c-bg-soft);
}

.steps-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2.5rem;
}

.step-card {
  display: flex;
  gap: 1rem;
  padding: 1.5rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
}

.step-number {
  font-size: 1.75rem;
  font-weight: 800;
  color: var(--vp-c-brand-1);
  line-height: 1;
  min-width: 2.5rem;
}

.step-title {
  font-size: 1rem;
  font-weight: 600;
  margin: 0 0 0.4rem;
  color: var(--vp-c-text-1);
}

.step-desc {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  margin: 0;
  line-height: 1.6;
}

.flow-arrow-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 0.25rem;
}

.flow-chip {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.8rem;
  font-weight: 600;
  padding: 0.3rem 0.75rem;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border-radius: 999px;
  white-space: nowrap;
}

.flow-arrow {
  background: none;
  padding: 0;
  color: var(--vp-c-text-3);
  font-size: 1rem;
  font-weight: 400;
  margin-left: 0.4rem;
}
</style>
