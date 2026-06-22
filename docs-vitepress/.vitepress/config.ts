import { defineConfig } from 'vitepress'

const GITHUB_PAGES_BASE = '/sklearn-genetic-opt/'

// Sidebar shared across versions — paths are relative to the version root
function versionSidebar(versionPrefix: string) {
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
        { text: 'Adapters', link: `${versionPrefix}/guide/adapters` },
        { text: 'Advanced Optimizer Control', link: `${versionPrefix}/guide/advanced-optimizer-control` },
        { text: 'MLflow Integration', link: `${versionPrefix}/guide/mlflow` },
        { text: 'Outlier Detection', link: `${versionPrefix}/guide/outliers` },
        { text: 'Reproducibility', link: `${versionPrefix}/guide/reproducibility` },
        { text: 'Troubleshooting', link: `${versionPrefix}/guide/troubleshooting` },
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

  srcExclude: ['**/CLAUDE.md', '**/README.md'],
  ignoreDeadLinks: [/\/related-page$/],

  head: [
    ['link', { rel: 'icon', href: `${GITHUB_PAGES_BASE}logo.png` }],
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
