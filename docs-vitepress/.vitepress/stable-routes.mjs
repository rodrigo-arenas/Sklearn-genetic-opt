// URL rules for the released documentation. The newest numbered release renders
// at /stable/, so its URLs stay the same when the next version is frozen; its
// numbered URLs become redirect stubs and every other copy of a stable page
// canonicalizes to /stable/<page>. Paths use forward slashes, as VitePress does.
import { existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

// Numbered release directories under versions/, newest first.
export function discoverReleaseVersions(docsRoot) {
  const versionsRoot = join(docsRoot, 'versions')
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

// VitePress `rewrites` function: versions/<stable>/<page>.md renders as stable/<page>.md.
export function stableRewrite(stableVersion) {
  const prefix = `versions/${stableVersion}/`
  return (page) => (stableVersion && page.startsWith(prefix) ? `stable/${page.slice(prefix.length)}` : page)
}

// /stable/ URL of a rendered older-release or `latest` page that the stable
// release also has; undefined for the stable pages themselves and for pages
// the stable release lacks.
export function stableCounterpart(docsRoot, stableVersion, relativePath) {
  const match = relativePath.replace(/\\/g, '/').match(/^versions\/(latest|\d+\.\d+)\/(.*)\.md$/)
  if (!match || !stableVersion) return undefined
  if (!existsSync(join(docsRoot, 'versions', stableVersion, `${match[2]}.md`))) return undefined
  return `/stable/${match[2]}`.replace(/\/index$/, '/')
}

// Static page that sends visitors and crawlers to `target`. It loads no
// VitePress app, whose router would render its 404 page for an unknown route.
export function writeRedirectStub(file, site, target) {
  mkdirSync(dirname(file), { recursive: true })
  writeFileSync(file, `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="robots" content="noindex,follow">
  <meta http-equiv="refresh" content="0; url=${target}">
  <link rel="canonical" href="${site}${target}">
  <title>Documentation moved | sklearn-genetic-opt</title>
</head>
<body><p>Continue to the <a href="${target}">updated documentation</a>.</p></body>
</html>
`)
}

// Keep the stable release's numbered URLs working: one stub per built
// stable/<page>.html at versions/<stable>/<page>.html, which the host also
// serves for the extensionless URL.
export function writeStableStubs(outDir, site, stableVersion) {
  if (!stableVersion) return
  const stableDir = join(outDir, 'stable')
  for (const entry of readdirSync(stableDir, { recursive: true })) {
    const page = entry.replace(/\\/g, '/')
    if (!page.endsWith('.html')) continue
    const target = `/stable/${page.slice(0, -'.html'.length).replace(/(^|\/)index$/, '$1')}`
    writeRedirectStub(join(outDir, 'versions', stableVersion, page), site, target)
  }
}
