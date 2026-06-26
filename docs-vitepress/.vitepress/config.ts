import { defineConfig, HeadConfig } from 'vitepress'

const GITHUB_PAGES_BASE = '/'
const GA_TAG = process.env.GOOGLE_ANALYTICS_TAG

const gaHead: HeadConfig[] = GA_TAG
  ? [
      ['script', { async: '', src: `https://www.googletagmanager.com/gtag/js?id=${GA_TAG}` }],
      ['script', {}, `window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${GA_TAG}');`],
    ]
  : []

// Sidebar shared across versions — paths are relative to the version root
function versionSidebar(versionPrefix: string) {
  // These sections and items only appear in the in-development (latest) docs.
  // New pages are only published in versions/latest/ until the next stable release.
  const isLatest = versionPrefix.endsWith('/latest')
  const benchmarksSection = isLatest
    ? [
        {
          text: 'Benchmarks',
          collapsed: false,
          items: [
            { text: 'Bayesmark Comparison', link: `${versionPrefix}/benchmarks/` },
          ],
        },
      ]
    : []
  const comparisonsSection = isLatest
    ? [
        {
          text: 'Comparisons',
          collapsed: false,
          items: [
            { text: 'Overview', link: `${versionPrefix}/comparisons/` },
            {
              text: 'Grid vs Random vs Bayesian vs Genetic',
              link: `${versionPrefix}/comparisons/grid-search-vs-genetic-algorithms`,
            },
            {
              text: 'Optuna vs sklearn-genetic-opt',
              link: `${versionPrefix}/comparisons/optuna-vs-sklearn-genetic-opt`,
            },
          ],
        },
      ]
    : []

  // New Getting Started items — only available in latest/
  const latestGettingStarted = isLatest
    ? [{ text: 'How Hyperparameter Optimization Works', link: `${versionPrefix}/guide/how-hyperparameter-optimization-works` }]
    : []

  // New User Guide items — only available in latest/
  const latestGuideItems = isLatest
    ? [
        { text: 'Choosing the Right Search Space',        link: `${versionPrefix}/guide/choosing-search-spaces` },
        { text: 'Common Hyperparameter Tuning Mistakes',  link: `${versionPrefix}/guide/common-mistakes` },
        { text: 'Feature Selection Methods Compared',     link: `${versionPrefix}/guide/feature-selection-guide` },
      ]
    : []

  // New Tutorial items — only available in latest/
  const latestTutorialItems = isLatest
    ? [
        { text: 'Random Forest Hyperparameter Tuning',          link: `${versionPrefix}/tutorials/tune-random-forest` },
        { text: 'Gradient Boosting Hyperparameter Tuning',      link: `${versionPrefix}/tutorials/tune-gradient-boosting` },
        { text: 'Logistic Regression Hyperparameter Tuning',    link: `${versionPrefix}/tutorials/tune-logistic-regression` },
        { text: 'SVM Hyperparameter Tuning (C, kernel, gamma)', link: `${versionPrefix}/tutorials/tune-svm` },
      ]
    : []

  return [
    {
      text: 'Getting Started',
      collapsed: false,
      items: [
        { text: 'Introduction', link: `${versionPrefix}/` },
        ...latestGettingStarted,
        { text: 'When to Use Genetic Algorithm Search', link: `${versionPrefix}/guide/when-to-use` },
        { text: 'Getting Started with GASearchCV', link: `${versionPrefix}/guide/basic-usage` },
        { text: 'Installation', link: `${versionPrefix}/guide/installation` },
      ],
    },
    {
      text: 'User Guide',
      collapsed: false,
      items: [
        ...latestGuideItems,
        { text: 'Cross-Validation in Hyperparameter Search', link: `${versionPrefix}/guide/understand-cv` },
        { text: 'Tuning scikit-learn Pipelines', link: `${versionPrefix}/guide/pipeline-tuning` },
        { text: 'Multi-Metric Optimization', link: `${versionPrefix}/guide/multi-metric` },
        { text: 'Early Stopping with Callbacks', link: `${versionPrefix}/guide/callbacks` },
        { text: 'Writing Custom Callbacks', link: `${versionPrefix}/guide/custom-callback` },
        { text: 'Adaptive Crossover & Mutation Schedules', link: `${versionPrefix}/guide/adapters` },
        { text: 'Advanced Optimizer Control', link: `${versionPrefix}/guide/advanced-optimizer-control` },
        { text: 'MLflow Integration', link: `${versionPrefix}/guide/mlflow` },
        { text: 'Tuning Outlier Detection Models', link: `${versionPrefix}/guide/outliers` },
        { text: 'Reproducibility & Checkpointing', link: `${versionPrefix}/guide/reproducibility` },
        { text: 'Troubleshooting', link: `${versionPrefix}/guide/troubleshooting` },
        { text: 'Migrating from 0.12', link: `${versionPrefix}/guide/migrating-from-0.12` },
      ],
    },
    {
      text: 'Tutorials',
      collapsed: false,
      items: [
        { text: 'Overview',                                       link: `${versionPrefix}/tutorials/` },
        ...latestTutorialItems,
        { text: 'XGBoost Hyperparameter Tuning',                  link: `${versionPrefix}/tutorials/tune-xgboost` },
        { text: 'LightGBM Hyperparameter Tuning',                 link: `${versionPrefix}/tutorials/tune-lightgbm` },
        { text: 'CatBoost Hyperparameter Tuning',                 link: `${versionPrefix}/tutorials/tune-catboost` },
        { text: 'Feature Selection with Genetic Algorithms',       link: `${versionPrefix}/tutorials/feature-selection` },
        { text: 'Hyperparameter Tuning for Imbalanced Datasets',  link: `${versionPrefix}/tutorials/imbalanced-classification` },
        { text: 'Isolation Forest Hyperparameter Tuning',         link: `${versionPrefix}/tutorials/isolation-forest` },
      ],
    },
    {
      text: 'Examples',
      collapsed: false,
      items: [
        { text: 'Overview',                               link: `${versionPrefix}/examples/` },
        { text: 'Grid Search vs Genetic Algorithms',      link: `${versionPrefix}/examples/sklearn-comparison` },
        { text: 'Random Forest: All Advanced Features',   link: `${versionPrefix}/examples/advanced-rf` },
        { text: 'Pipeline Regression',                    link: `${versionPrefix}/examples/pipeline-regression` },
        { text: 'Feature Selection',                      link: `${versionPrefix}/examples/feature-selection` },
        { text: 'Multi-Metric Search',                    link: `${versionPrefix}/examples/multi-metric` },
        { text: 'MLflow Experiment Tracking',             link: `${versionPrefix}/examples/mlflow-tracking` },
        { text: 'Checkpointing & Resume',                 link: `${versionPrefix}/examples/checkpointing` },
        { text: 'Plotting Gallery',                       link: `${versionPrefix}/examples/plotting-gallery` },
      ],
    },
    {
      text: 'API Reference',
      collapsed: false,
      items: [
        { text: 'GASearchCV', link: `${versionPrefix}/api/gasearchcv` },
        { text: 'GAFeatureSelectionCV', link: `${versionPrefix}/api/gafeatureselectioncv` },
        { text: 'Config Objects', link: `${versionPrefix}/api/config` },
        { text: 'Callbacks', link: `${versionPrefix}/api/callbacks` },
        { text: 'Schedules', link: `${versionPrefix}/api/schedules` },
        { text: 'Plots', link: `${versionPrefix}/api/plots` },
        { text: 'MLflow', link: `${versionPrefix}/api/mlflow` },
        { text: 'Search Space', link: `${versionPrefix}/api/space` },
        { text: 'Algorithms', link: `${versionPrefix}/api/algorithms` },
      ],
    },
    ...benchmarksSection,
    ...comparisonsSection,
    {
      text: 'Release Notes',
      collapsed: false,
      items: [
        { text: 'Changelog', link: `${versionPrefix}/release-notes` },
      ],
    },
  ]
}

export default defineConfig({
  title: 'sklearn-genetic-opt',
  description: 'Evolutionary hyperparameter tuning and feature selection for scikit-learn',
  base: GITHUB_PAGES_BASE,

  titleTemplate: ':title | sklearn-genetic-opt',

  sitemap: {
    hostname: 'https://sklearngeneticopt.rodrigo-arenas.com',
  },

  srcExclude: ['**/CLAUDE.md', '**/README.md'],

  transformPageData(pageData) {
    const base = 'https://sklearngeneticopt.rodrigo-arenas.com'
    const ogImage = `${base}/sklearn-genetic-opt-logo-128.png`
    const pageTitle = pageData.title
      ? `${pageData.title} | sklearn-genetic-opt`
      : 'sklearn-genetic-opt'
    const pageDescription =
      pageData.description ||
      'Evolutionary hyperparameter tuning and feature selection for scikit-learn'
    const cleanPath = pageData.relativePath
      .replace(/\\/g, '/')
      .replace(/index\.md$/, '')
      .replace(/\.md$/, '.html')
    const pageUrl = `${base}/${cleanPath}`

    pageData.frontmatter.head ??= []
    pageData.frontmatter.head.push(
      ['link', { rel: 'canonical',           href: pageUrl }],
      ['meta', { property: 'og:type',        content: 'website' }],
      ['meta', { property: 'og:title',       content: pageTitle }],
      ['meta', { property: 'og:description', content: pageDescription }],
      ['meta', { property: 'og:url',         content: pageUrl }],
      ['meta', { property: 'og:image',       content: ogImage }],
      ['meta', { name: 'twitter:card',        content: 'summary' }],
      ['meta', { name: 'twitter:title',       content: pageTitle }],
      ['meta', { name: 'twitter:description', content: pageDescription }],
    )

    if (pageData.relativePath === 'index.md') {
      pageData.frontmatter.head.push([
        'script',
        { type: 'application/ld+json' },
        JSON.stringify({
          '@context': 'https://schema.org',
          '@type': 'SoftwareApplication',
          name: 'sklearn-genetic-opt',
          description:
            'Evolutionary hyperparameter tuning and feature selection for scikit-learn using genetic algorithms powered by DEAP.',
          applicationCategory: 'DeveloperApplication',
          operatingSystem: 'Linux, macOS, Windows',
          programmingLanguage: 'Python',
          url: `${base}/`,
          license: 'https://opensource.org/licenses/MIT',
          author: { '@type': 'Person', name: 'Rodrigo Arenas Gómez' },
          offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
        }),
      ])
    }
  },

  head: [
    ['link', { rel: 'icon', href: `${GITHUB_PAGES_BASE}logo.png` }],
    ...gaHead,
  ],

  themeConfig: {
    logo: '/logo.png',
    siteTitle: 'sklearn-genetic-opt',

    nav: [
      { text: 'Home', link: '/' },
      {
        text: 'Version',
        items: [
          { text: '0.13 (stable)', link: '/versions/0.13/' },
          { text: 'latest (dev)', link: '/versions/latest/' },
        ],
      },
      {
        text: 'Links',
        items: [
          { text: 'GitHub', link: 'https://github.com/rodrigo-arenas/Sklearn-genetic-opt' },
          { text: 'PyPI', link: 'https://pypi.org/project/sklearn-genetic-opt/' },
          { text: 'Legacy Docs (RTD)', link: 'https://sklearn-genetic-opt.readthedocs.io' },
        ],
      },
    ],

    sidebar: {
      '/versions/0.13/': versionSidebar('/versions/0.13'),
      '/versions/latest/': versionSidebar('/versions/latest'),
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/rodrigo-arenas/Sklearn-genetic-opt' },
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2021–present Rodrigo Arenas Gómez',
    },

    search: {
      provider: 'local',
    },

    editLink: {
      pattern: 'https://github.com/rodrigo-arenas/Sklearn-genetic-opt/edit/master/docs-vitepress/:path',
      text: 'Edit this page on GitHub',
    },
  },
})
