<template>
  <section class="home-section home-why-ga">
    <div class="home-container">
      <h2 class="section-title">Why Evolutionary Algorithms?</h2>
      <p class="section-subtitle">
        Each method has strengths. Genetic algorithms win on large search spaces with parameter interactions.
      </p>

      <div class="table-wrapper">
        <table class="comparison-table">
          <thead>
            <tr>
              <th>Method</th>
              <th>Handles Interactions</th>
              <th>Scales to 10+ Params</th>
              <th>sklearn Compatible</th>
              <th>Feature Selection</th>
              <th>Best For</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in rows" :key="row.method" :class="{ highlight: row.highlight }">
              <td class="method-col">
                <span class="method-name">{{ row.method }}</span>
              </td>
              <td class="center"><span :class="iconClass(row.interactions)">{{ iconText(row.interactions) }}</span></td>
              <td class="center"><span :class="iconClass(row.scales)">{{ iconText(row.scales) }}</span></td>
              <td class="center"><span :class="iconClass(row.sklearn)">{{ iconText(row.sklearn) }}</span></td>
              <td class="center"><span :class="iconClass(row.featuresel)">{{ iconText(row.featuresel) }}</span></td>
              <td class="use-case">{{ row.bestFor }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p class="table-note">
        ✦ = sklearn-genetic-opt &nbsp;|&nbsp; ✓ = yes &nbsp;|&nbsp; ~ = partial &nbsp;|&nbsp; ✗ = no
      </p>
    </div>
  </section>
</template>

<script setup>
const rows = [
  {
    method: 'GridSearchCV',
    interactions: 'partial',
    scales: false,
    sklearn: true,
    featuresel: false,
    bestFor: '< 4 parameters, exhaustive coverage needed',
  },
  {
    method: 'RandomizedSearchCV',
    interactions: 'partial',
    scales: true,
    sklearn: true,
    featuresel: false,
    bestFor: 'Quick baseline, budget constrained',
  },
  {
    method: 'Optuna',
    interactions: true,
    scales: true,
    sklearn: 'partial',
    featuresel: false,
    bestFor: 'Bayesian search, non-sklearn objectives',
  },
  {
    method: 'RFE / SelectFromModel',
    interactions: false,
    scales: 'partial',
    sklearn: true,
    featuresel: true,
    bestFor: 'Feature selection only, no hyperparameter tuning',
  },
  {
    method: 'sklearn-genetic-opt ✦',
    interactions: true,
    scales: true,
    sklearn: true,
    featuresel: true,
    bestFor: 'Joint hyperparameter + feature search in one step',
    highlight: true,
  },
]

function iconText(val) {
  if (val === true) return '✓'
  if (val === false) return '✗'
  return '~'
}

function iconClass(val) {
  if (val === true) return 'icon-yes'
  if (val === false) return 'icon-no'
  return 'icon-partial'
}
</script>

<style scoped>
.home-why-ga {
  background: var(--vp-c-bg);
}

.table-wrapper {
  overflow-x: auto;
  border-radius: 10px;
  border: 1px solid var(--vp-c-divider);
  margin-bottom: 1rem;
}

.comparison-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.comparison-table thead tr {
  background: var(--vp-c-bg-soft);
}

.comparison-table th {
  padding: 0.75rem 1rem;
  text-align: left;
  font-weight: 600;
  color: var(--vp-c-text-1);
  border-bottom: 1px solid var(--vp-c-divider);
  white-space: nowrap;
}

.comparison-table td {
  padding: 0.7rem 1rem;
  border-bottom: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-2);
}

.comparison-table tr:last-child td {
  border-bottom: none;
}

.comparison-table tr.highlight {
  background: var(--vp-c-brand-soft);
}

.comparison-table tr.highlight td {
  color: var(--vp-c-text-1);
}

.method-name {
  font-weight: 600;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 0.8rem;
}

.center {
  text-align: center;
}

.icon-yes {
  color: #22c55e;
  font-weight: 700;
}

.icon-no {
  color: var(--vp-c-text-3);
}

.icon-partial {
  color: #f59e0b;
  font-weight: 700;
}

.use-case {
  font-size: 0.8rem;
  min-width: 200px;
}

.table-note {
  font-size: 0.78rem;
  color: var(--vp-c-text-3);
  text-align: center;
  margin: 0;
}
</style>
