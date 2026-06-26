#!/usr/bin/env node
/**
 * Freeze the latest/ docs as a numbered stable version.
 *
 * Usage:   node scripts/freeze-version.mjs <version>
 * Example: node scripts/freeze-version.mjs 0.14
 *
 * What it does:
 *   1. Copies versions/latest/ → versions/<version>/
 *   2. Strips the :::warning Development version ::: banner from all pages
 *   3. Updates "X.Y (stable)" labels in :::info Version boxes to the new number
 *   4. Adds the new version to the nav dropdown and sidebar in config.ts
 *   5. Updates versions/stable/index.md to redirect to the new stable version
 *   6. Updates "see [0.X]" references inside latest/ dev banners to point to
 *      the new stable version
 */

import { cpSync, rmSync, readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const docsRoot = join(__dirname, '..')

const version = process.argv[2]
if (!version || !/^\d+\.\d+$/.test(version)) {
  console.error('Usage: node scripts/freeze-version.mjs <version>')
  console.error('Example: node scripts/freeze-version.mjs 0.14')
  process.exit(1)
}

const latestDir = join(docsRoot, 'versions', 'latest')
const frozenDir = join(docsRoot, 'versions', version)

// ── 1 & 2 & 3. Copy latest/ → versions/<version>/ then clean it up ──────────
// Always overwrite: patch releases (e.g. 0.13.1) reuse the same minor folder
// (e.g. 0.13/) so we delete and re-copy to avoid stale files.
{
  if (existsSync(frozenDir)) {
    console.log(`versions/${version}/ already exists — overwriting for patch release.`)
    rmSync(frozenDir, { recursive: true, force: true })
  }
  console.log(`Copying versions/latest/ → versions/${version}/`)
  cpSync(latestDir, frozenDir, { recursive: true })

  walkMd(frozenDir, (filePath) => {
    let content = readFileSync(filePath, 'utf8')

    // Strip the :::warning Development version … ::: block (and surrounding blank lines)
    content = content.replace(/\n?:::warning Development version[\s\S]*?:::\n*/g, '\n')

    // Fix "X.Y (stable)" in any :::info Version box to use the new version number
    content = content.replace(/\*\*[\d.]+ \(stable\)\*\*/g, `**${version} (stable)**`)

    writeFileSync(filePath, content, 'utf8')
  })

  console.log(`Cleaned up versions/${version}/`)
}

// ── 4. Update config.ts ───────────────────────────────────────────────────────
const configPath = join(docsRoot, '.vitepress', 'config.ts')
let config = readFileSync(configPath, 'utf8')

if (config.includes(`'/versions/${version}/'`)) {
  console.log(`config.ts already references ${version} — skipping config update.`)
} else {
  const lines = config.split('\n')

  // 4a. Demote the current stable in the nav: "X.Y (stable)" → "X.Y"
  for (let i = 0; i < lines.length; i++) {
    lines[i] = lines[i].replace(/(\d+\.\d+) \(stable\)(')/g, '$1$2')
  }

  // 4b. Insert new stable as the first entry in the Version dropdown items: [...]
  let inVersionDropdown = false
  let itemsLine = -1
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes("text: 'Version',")) inVersionDropdown = true
    if (inVersionDropdown && lines[i].trimStart().startsWith('items: [')) {
      itemsLine = i
      break
    }
  }
  if (itemsLine === -1) throw new Error("Could not find Version dropdown 'items: [' in config.ts")

  const navIndent = lines[itemsLine + 1].match(/^(\s*)/)[1]
  lines.splice(
    itemsLine + 1,
    0,
    `${navIndent}{ text: '${version} (stable)', link: '/versions/${version}/' },`
  )

  // 4c. Add sidebar entry for the new version (before '/versions/latest/')
  const latestSidebarEntry = `'/versions/latest/': versionSidebar('/versions/latest')`
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes(latestSidebarEntry)) {
      const sidebarIndent = lines[i].match(/^(\s*)/)[1]
      lines.splice(
        i,
        0,
        `${sidebarIndent}'/versions/${version}/': versionSidebar('/versions/${version}'),`
      )
      break
    }
  }

  config = lines.join('\n')
  writeFileSync(configPath, config, 'utf8')
  console.log(`Updated config.ts: added version ${version} to nav and sidebar`)
}

// ── 5. Update /stable redirect to point at the new stable version ────────────
const stablePage = join(docsRoot, 'versions', 'stable', 'index.md')
if (existsSync(stablePage)) {
  let stableContent = readFileSync(stablePage, 'utf8')
  stableContent = stableContent.replace(
    /url:\/versions\/[\d.]+\//g,
    `url:/versions/${version}/`
  )
  stableContent = stableContent.replace(
    /window\.location\.replace\('\/versions\/[\d.]+\/'\)/g,
    `window.location.replace('/versions/${version}/')`
  )
  writeFileSync(stablePage, stableContent, 'utf8')
  console.log(`Updated /stable redirect → /versions/${version}/`)
}

// ── 6. Update "see [0.X]" references in latest/ dev banners ─────────────────
walkMd(latestDir, (filePath) => {
  let content = readFileSync(filePath, 'utf8')
  // Replace version links inside the warning block only
  const updated = content.replace(
    /(:::warning Development version[\s\S]*?:::)/g,
    (banner) =>
      banner.replace(
        /\[[\d.]+(?:\s+\([^)]+\))?\]\(\/versions\/[\d.]+\/\)/g,
        `[${version}](/versions/${version}/)`
      )
  )
  if (updated !== content) writeFileSync(filePath, updated, 'utf8')
})
console.log(`Updated latest/ dev banners to reference version ${version}`)

console.log(`\nDone — version ${version} is now frozen as stable.`)

// ── Helpers ───────────────────────────────────────────────────────────────────
function walkMd(dir, fn) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) walkMd(full, fn)
    else if (entry.endsWith('.md')) fn(full)
  }
}
