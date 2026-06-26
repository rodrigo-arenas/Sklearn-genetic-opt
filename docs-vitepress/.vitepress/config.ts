import { defineConfig, HeadConfig } from 'vitepress'

const GITHUB_PAGES_BASE = '/Sklearn-genetic-opt/'
const GA_TAG = process.env.GOOGLE_ANALYTICS_TAG

const gaHead: HeadConfig[] = GA_TAG
  ? [
      ['script', { async: '', src: `https://www.googletagmanager.com/gtag/js?id=${GA_TAG}` }],
      ['script', {}, `window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${GA_TAG}');`],
    ]
  : []

// Sidebar shared across versions — paths are relative to the version root
function versionSidebar(versionPrefix: string) {
  // The Benchmarks section is only published for the in-development docs.
  const isLatest = versionPrefix.endsWith('/latest')
  const benchmarksSection = isLatest
    ? [
        {
          text: 'Benchmarks',
          items: [
            { text: 'Bayesmark Comparison', link: `${versionPrefix}/benchmarks/` },
          ],
        },
      ]
    : []

  return [
    {
      text: 'Getting Started',
      items: [
        { text: 'Introduction', link: `${versionPrefix}/` },
        { text: 'When to Use', link: `${versionPrefix}/guide/when-to-use` },
        { text: 'Basic Usage', link: `${versionPrefix}/guide/basic-usage` },
        { text: 'Installation', link: `${versionPrefix}/guide/installation` },
      ],
    },
    {
      text: 'User Guide',
      items: [
        { text: 'Understanding Cross-Validation', link: `${versionPrefix}/guide/understand-cv` },
        { text: 'Pipeline Tuning', link: `${versionPrefix}/guide/pipeline-tuning` },
        { text: 'Multi-Metric Optimization', link: `${versionPrefix}/guide/multi-metric` },
        { text: 'Callbacks', link: `${versionPrefix}/guide/callbacks` },
        { text: 'Custom Callbacks', link: `${versionPrefix}/guide/custom-callback` },
        { text: 'Adaptive Schedules', link: `${versionPrefix}/guide/adapters` },
        { text: 'Advanced Optimizer Control', link: `${versionPrefix}/guide/advanced-optimizer-control` },
        { text: 'MLflow Integration', link: `${versionPrefix}/guide/mlflow` },
        { text: 'Outlier Detection', link: `${versionPrefix}/guide/outliers` },
        { text: 'Reproducibility', link: `${versionPrefix}/guide/reproducibility` },
        { text: 'Troubleshooting', link: `${versionPrefix}/guide/troubleshooting` },
        { text: 'Migrating from 0.12', link: `${versionPrefix}/guide/migrating-from-0.12` },
      ],
    },
    {
      text: 'Tutorials',
      items: [
        { text: 'Overview',                        link: `${versionPrefix}/tutorials/` },
        { text: 'Tune XGBoost',                    link: `${versionPrefix}/tutorials/tune-xgboost` },
        { text: 'Tune LightGBM',                   link: `${versionPrefix}/tutorials/tune-lightgbm` },
        { text: 'Tune CatBoost',                   link: `${versionPrefix}/tutorials/tune-catboost` },
        { text: 'Comprehensive Feature Selection',  link: `${versionPrefix}/tutorials/feature-selection` },
        { text: 'Imbalanced Classification',        link: `${versionPrefix}/tutorials/imbalanced-classification` },
        { text: 'Isolation Forest',                 link: `${versionPrefix}/tutorials/isolation-forest` },
      ],
    },
    {
      text: 'API Reference',
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
    {
      text: 'Examples',
      items: [
        { text: 'Overview', link: `${versionPrefix}/examples/` },
        { text: 'Comparing Search Methods', link: `${versionPrefix}/examples/sklearn-comparison` },
        { text: 'Advanced Random Forest', link: `${versionPrefix}/examples/advanced-rf` },
        { text: 'Pipeline Regression', link: `${versionPrefix}/examples/pipeline-regression` },
        { text: 'Feature Selection (Noisy Data)', link: `${versionPrefix}/examples/feature-selection` },
        { text: 'Multi-Metric Search', link: `${versionPrefix}/examples/multi-metric` },
        { text: 'MLflow Tracking', link: `${versionPrefix}/examples/mlflow-tracking` },
        { text: 'Checkpointing', link: `${versionPrefix}/examples/checkpointing` },
        { text: 'Plotting Gallery', link: `${versionPrefix}/examples/plotting-gallery` },
      ],
    },
    ...benchmarksSection,
    {
      text: 'Release Notes',
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
    hostname: 'https://rodrigo-arenas.github.io',
  },

  srcExclude: ['**/CLAUDE.md', '**/README.md'],

  transformPageData(pageData) {
    const base = 'https://rodrigo-arenas.github.io/Sklearn-genetic-opt'
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
