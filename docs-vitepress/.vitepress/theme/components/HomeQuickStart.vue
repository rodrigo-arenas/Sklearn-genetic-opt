<template>
  <section class="home-section home-quick-start">
    <div class="home-container">
      <div class="qs-layout">
        <div class="qs-left">
          <h2 class="section-title left-align">Up and Running in 30 Seconds</h2>
          <p class="section-subtitle left-align">
            Install the package, define a search space, call <code>fit</code>.
            The GA finds better hyperparameters than a grid search in the same budget.
          </p>

          <div class="qs-steps">
            <div class="qs-step">
              <span class="qs-step-num">1</span>
              <span>Install</span>
            </div>
            <div class="qs-step">
              <span class="qs-step-num">2</span>
              <span>Define search space</span>
            </div>
            <div class="qs-step">
              <span class="qs-step-num">3</span>
              <span>Fit &amp; inspect results</span>
            </div>
          </div>

          <div class="qs-ctas">
            <a href="/stable/" class="qs-btn primary">Full Installation Guide</a>
            <a href="/versions/latest/tutorials/" class="qs-btn secondary">All Tutorials</a>
          </div>
        </div>

        <div class="qs-right">
          <div class="code-block">
            <div class="code-header">
              <span class="code-dot red"></span>
              <span class="code-dot yellow"></span>
              <span class="code-dot green"></span>
              <span class="code-lang">Python</span>
            </div>
            <pre class="code-body"><code><span class="c"># pip install sklearn-genetic-opt</span>

<span class="kw">from</span> sklearn.datasets <span class="kw">import</span> load_breast_cancer
<span class="kw">from</span> sklearn.ensemble <span class="kw">import</span> RandomForestClassifier
<span class="kw">from</span> sklearn_genetic <span class="kw">import</span> GASearchCV
<span class="kw">from</span> sklearn_genetic.space <span class="kw">import</span> Integer, Continuous

X, y = load_breast_cancer(return_X_y=<span class="kw">True</span>)

param_grid = {
    <span class="s">"n_estimators"</span>:     Integer(<span class="n">50</span>, <span class="n">500</span>),
    <span class="s">"max_depth"</span>:        Integer(<span class="n">3</span>, <span class="n">15</span>),
    <span class="s">"min_samples_split"</span>: Integer(<span class="n">2</span>, <span class="n">20</span>),
    <span class="s">"max_features"</span>:     Continuous(<span class="n">0.2</span>, <span class="n">1.0</span>),
}

evolved_estimator = GASearchCV(
    estimator=RandomForestClassifier(),
    cv=<span class="n">5</span>,
    param_grid=param_grid,
    population_size=<span class="n">20</span>,
    generations=<span class="n">15</span>,
    random_state=<span class="n">42</span>,
    n_jobs=<span class="n">-1</span>,
)

evolved_estimator.fit(X, y)
<span class="nb">print</span>(evolved_estimator.best_params_)
<span class="nb">print</span>(evolved_estimator.best_score_)</code></pre>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.home-quick-start {
  background: var(--vp-c-bg-soft);
}

.qs-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3rem;
  align-items: center;
}

@media (max-width: 768px) {
  .qs-layout {
    grid-template-columns: 1fr;
    gap: 2rem;
  }
}

.left-align {
  text-align: left;
}

.left-align.section-subtitle {
  max-width: none;
}

.qs-steps {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin: 1.5rem 0;
}

.qs-step {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.95rem;
  color: var(--vp-c-text-2);
}

.qs-step-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.6rem;
  height: 1.6rem;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  color: white;
  font-size: 0.75rem;
  font-weight: 700;
  flex-shrink: 0;
}

.qs-ctas {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.qs-btn {
  padding: 0.6rem 1.25rem;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 600;
  text-decoration: none;
  transition: opacity 0.15s;
}

.qs-btn:hover {
  opacity: 0.85;
}

.qs-btn.primary {
  background: var(--vp-c-brand-1);
  color: white;
}

.qs-btn.secondary {
  border: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-1);
}

/* Code block */
.code-block {
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  font-size: 0.82rem;
}

.code-header {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.6rem 1rem;
  background: #1e2030;
}

.code-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.code-dot.red    { background: #ff5f57; }
.code-dot.yellow { background: #febc2e; }
.code-dot.green  { background: #28c840; }

.code-lang {
  margin-left: auto;
  font-size: 0.7rem;
  color: #6b7280;
  font-family: var(--vp-font-family-mono);
}

.code-body {
  margin: 0;
  padding: 1.25rem;
  background: #1e2030;
  overflow-x: auto;
  line-height: 1.65;
}

.code-body code {
  font-family: var(--vp-font-family-mono);
  font-size: 0.82rem;
  color: #cdd6f4;
}

.c  { color: #6c7086; }
.kw { color: #cba6f7; }
.s  { color: #a6e3a1; }
.n  { color: #fab387; }
.nb { color: #89dceb; }
</style>
