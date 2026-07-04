<template>
  <section class="home-section home-community">
    <div class="home-container">
      <div class="community-inner">
        <h2 class="community-title">Open Source &amp; Community Driven</h2>
        <p class="community-desc">
          sklearn-genetic-opt is MIT licensed and actively maintained. Contributions, bug reports,
          and feature requests are welcome. If it saves you time, a GitHub star helps other
          practitioners discover it.
        </p>

        <div class="community-stats">
          <div v-for="stat in stats" :key="stat.label" class="stat-item">
            <span class="stat-value">{{ stat.value }}</span>
            <span class="stat-label">{{ stat.label }}</span>
          </div>
        </div>

        <div class="community-ctas">
          <a
            href="https://github.com/rodrigo-arenas/Sklearn-genetic-opt"
            class="community-btn primary"
          >
            ⭐ Star on GitHub
          </a>
          <a
            href="https://github.com/rodrigo-arenas/Sklearn-genetic-opt/issues"
            class="community-btn secondary"
          >
            Report an Issue
          </a>
          <a
            href="https://github.com/rodrigo-arenas/Sklearn-genetic-opt/blob/master/CONTRIBUTING.md"
            class="community-btn secondary"
          >
            Contributing Guide
          </a>
        </div>

        <div class="install-snippet">
          <code>pip install sklearn-genetic-opt</code>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const stats = [
  { value: 'MIT', label: 'License' },
  { value: 'Python 3.12+', label: 'Requirement' },
  { value: 'scikit-learn', label: 'Compatible with' },
  { value: '0.13', label: 'Latest stable' },
]
</script>

<style scoped>
.home-community {
  background: var(--vp-c-bg);
  border-top: 1px solid var(--vp-c-divider);
}

.community-inner {
  text-align: center;
  max-width: 680px;
  margin: 0 auto;
}

.community-title {
  font-size: 1.75rem;
  font-weight: 800;
  color: var(--vp-c-text-1);
  margin: 0 0 0.75rem;
  line-height: 1.2;
}

.community-desc {
  font-size: 0.95rem;
  color: var(--vp-c-text-2);
  line-height: 1.65;
  margin: 0 0 2rem;
}

.community-stats {
  display: flex;
  justify-content: center;
  gap: 2.5rem;
  flex-wrap: wrap;
  margin-bottom: 2rem;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.stat-value {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  font-family: var(--vp-font-family-mono);
}

.stat-label {
  font-size: 0.78rem;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.community-ctas {
  display: flex;
  justify-content: center;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-bottom: 2rem;
}

.community-btn {
  padding: 0.65rem 1.4rem;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 600;
  text-decoration: none;
  transition: opacity 0.15s;
}

.community-btn:hover {
  opacity: 0.85;
}

.community-btn.primary {
  background: var(--vp-c-brand-1);
  color: white;
}

.community-btn.secondary {
  border: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-1);
}

.install-snippet {
  display: inline-block;
  padding: 0.6rem 1.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
}

.install-snippet code {
  font-family: var(--vp-font-family-mono);
  font-size: 0.9rem;
  color: var(--vp-c-text-1);
}
</style>
