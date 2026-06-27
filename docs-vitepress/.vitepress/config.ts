import { existsSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, HeadConfig } from 'vitepress'

const GITHUB_PAGES_BASE = '/'
const GA_TAG = process.env.GOOGLE_ANALYTICS_TAG
const DOCS_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')

function discoverReleaseVersions() {
  const versionsRoot = join(DOCS_ROOT, 'versions')
  if (!existsSync(versionsRoot)) return []

  return readdirSync(versionsRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^\d+\.\d+$/.test(entry.name))
    .map((entry) => entry.name)
    .sort((a, b) => {
      const [aMajor, aMinor] = a.split('.').map(Number)
      const [bMajor, bMinor] = b.split('.').map(Number)
      return bMajor - aMajor || bMinor - aMinor
    })
}

const releaseVersions = discoverReleaseVersions()

const versionNavItems = [
  ...releaseVersions.map((version, index) => ({
    text: index === 0 ? `${version} (stable)` : version,
    link: `/versions/${version}/`,
  })),
  { text: 'latest (dev)', link: '/versions/latest/' },
]

const versionSidebarEntries = Object.fromEntries([
  ...releaseVersions.map((version) => [
    `/versions/${version}/`,
    versionSidebar(`/versions/${version}`),
  ]),
  ['/versions/latest/', versionSidebar('/versions/latest')],
])

const gaHead: HeadConfig[] = GA_TAG
  ? [
      ['script', { async: '', src: `https://www.googletagmanager.com/gtag/js?id=${GA_TAG}` }],
      ['script', {}, `window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${GA_TAG}');`],
    ]
  : []

function versionName(versionPrefix: string) {
  return versionPrefix.replace(/\/$/, '').split('/').at(-1) || versionPrefix
}

function hasVersionPage(versionPrefix: string, relativePath: string) {
  const version = versionName(versionPrefix)
  const cleanPath = relativePath.replace(/^\/+/, '').replace(/\/$/, '/index')
  return existsSync(join(DOCS_ROOT, 'versions', version, `${cleanPath}.md`))
}

function pageItem(versionPrefix: string, text: string, relativePath: string) {
  return hasVersionPage(versionPrefix, relativePath)
    ? [{ text, link: `${versionPrefix}/${relativePath}` }]
    : []
}

function pageSection(
  versionPrefix: string,
  text: string,
  relativeRoot: string,
  items: Array<{ text: string; relativePath: string }>,
  collapsed = false,
) {
  const existingItems = items.flatMap((item) =>
    pageItem(versionPrefix, item.text, item.relativePath),
  )
  return hasVersionPage(versionPrefix, relativeRoot) || existingItems.length
    ? [{ text, collapsed, items: existingItems }]
    : []
}

// Sidebar shared across versions. Sections are included only when that version
// actually has the referenced pages, so frozen versions stay self-contained.
function versionSidebar(versionPrefix: string) {
  return [
    {
      text: 'Getting Started',
      collapsed: false,
      items: [
        { text: 'Introduction', link: `${versionPrefix}/` },
        ...pageItem(versionPrefix, 'How Hyperparameter Optimization Works', 'guide/how-hyperparameter-optimization-works'),
        { text: 'When to Use Genetic Algorithm Search', link: `${versionPrefix}/guide/when-to-use` },
        { text: 'Getting Started with GASearchCV', link: `${versionPrefix}/guide/basic-usage` },
        { text: 'Installation', link: `${versionPrefix}/guide/installation` },
      ],
    },
    {
      text: 'User Guide',
      collapsed: false,
      items: [
        ...pageItem(versionPrefix, 'Choosing the Right Search Space', 'guide/choosing-search-spaces'),
        ...pageItem(versionPrefix, 'Common Hyperparameter Tuning Mistakes', 'guide/common-mistakes'),
        ...pageItem(versionPrefix, 'Feature Selection Methods Compared', 'guide/feature-selection-guide'),
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
        ...pageItem(versionPrefix, 'Random Forest Hyperparameter Tuning', 'tutorials/tune-random-forest'),
        ...pageItem(versionPrefix, 'Gradient Boosting Hyperparameter Tuning', 'tutorials/tune-gradient-boosting'),
        ...pageItem(versionPrefix, 'Logistic Regression Hyperparameter Tuning', 'tutorials/tune-logistic-regression'),
        ...pageItem(versionPrefix, 'SVM Hyperparameter Tuning (C, kernel, gamma)', 'tutorials/tune-svm'),
        { text: 'XGBoost Hyperparameter Tuning',                  link: `${versionPrefix}/tutorials/tune-xgboost` },
        { text: 'LightGBM Hyperparameter Tuning',                 link: `${versionPrefix}/tutorials/tune-lightgbm` },
        { text: 'CatBoost Hyperparameter Tuning',                 link: `${versionPrefix}/tutorials/tune-catboost` },
        { text: 'Feature Selection with Genetic Algorithms',       link: `${versionPrefix}/tutorials/feature-selection` },
        { text: 'Hyperparameter Tuning for Imbalanced Datasets',  link: `${versionPrefix}/tutorials/imbalanced-classification` },
        { text: 'Isolation Forest Hyperparameter Tuning',         link: `${versionPrefix}/tutorials/isolation-forest` },
      ],
    },
    ...pageSection(versionPrefix, 'Recipes', 'recipes/', [
      { text: 'Overview', relativePath: 'recipes/' },
      { text: 'Classification', relativePath: 'recipes/classification/' },
      { text: 'Regression', relativePath: 'recipes/regression/' },
      { text: 'Feature Selection', relativePath: 'recipes/feature-selection/' },
      { text: 'Pipelines', relativePath: 'recipes/pipelines/' },
      { text: 'Scoring Metrics', relativePath: 'recipes/metrics/' },
      { text: 'Integrations', relativePath: 'recipes/integrations/' },
      { text: 'Advanced', relativePath: 'recipes/advanced/' },
    ]),
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
        ...pageItem(versionPrefix, 'Estimator Presets', 'api/presets'),
        { text: 'Algorithms', link: `${versionPrefix}/api/algorithms` },
      ],
    },
    ...pageSection(versionPrefix, 'Benchmarks', 'benchmarks/', [
      { text: 'Bayesmark Comparison', relativePath: 'benchmarks/' },
    ]),
    ...pageSection(versionPrefix, 'Comparisons', 'comparisons/', [
      { text: 'Overview', relativePath: 'comparisons/' },
      {
        text: 'Grid vs Random vs Bayesian vs Genetic',
        relativePath: 'comparisons/grid-search-vs-genetic-algorithms',
      },
      {
        text: 'Optuna vs sklearn-genetic-opt',
        relativePath: 'comparisons/optuna-vs-sklearn-genetic-opt',
      },
    ]),
    {
      text: 'Release Notes',
      collapsed: false,
      items: [
        ...pageItem(versionPrefix, 'Changelog', 'release-notes'),
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
        items: versionNavItems,
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

    sidebar: versionSidebarEntries,

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
