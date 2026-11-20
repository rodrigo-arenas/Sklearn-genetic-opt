<script setup>
const columns = [
  { a: 1, b: 0, parent: 'A' },
  { a: 0, b: 1, parent: 'B' },
  { a: 0, b: 1, parent: 'A' },
  { a: 1, b: 1, parent: 'B' },
  { a: 0, b: 1, parent: 'B' },
  { a: 1, b: 0, parent: 'A' },
]

const parents = [
  { name: 'A', key: 'a', y: 43 },
  { name: 'B', key: 'b', y: 87 },
]
const x = (index) => 111 + index * 49
</script>

<template>
  <figure class="crossover">
    <p class="crossover-heading">How crossover makes a candidate</p>
    <svg class="crossover-diagram" viewBox="0 0 396 219" role="img" aria-labelledby="crossover-title crossover-desc">
      <title id="crossover-title">Uniform crossover of two feature masks</title>
      <desc id="crossover-desc">Parent A has bits 1 0 0 1 0 1, and parent B has bits 0 1 1 1 1 0. One choice at each position, A B A B B A, produces an offspring with bits 1 1 0 1 1 1.</desc>

      <text x="1" y="17" class="row-label">Column</text>
      <text v-for="(_, i) in columns" :key="i" :x="x(i) + 18" y="17" class="column-label">{{ i + 1 }}</text>

      <g v-for="row in parents" :key="row.name">
        <text x="1" :y="row.y" class="row-label">Parent {{ row.name }}</text>
        <g
          v-for="(col, i) in columns"
          :key="i"
          :class="{ inherited: col.parent === row.name, 'from-b': row.name === 'B' }"
        >
          <rect :x="x(i)" :y="row.y - 19" width="36" height="32" class="gene" />
          <text :x="x(i) + 18" :y="row.y" class="bit">{{ col[row.key] }}</text>
        </g>
      </g>

      <path d="M1 116h390" class="row-rule" />
      <text x="1" y="142" class="row-label">Inherit from</text>
      <text x="1" y="198" class="row-label offspring-label">Offspring</text>
      <g v-for="(col, i) in columns" :key="i" :class="{ 'from-b': col.parent === 'B' }" class="inherited">
        <text :x="x(i) + 18" y="142" class="parent-label">{{ col.parent }}</text>
        <path :d="`M${x(i) + 18} 151v18m-4-4 4 4 4-4`" class="inheritance-line" />
        <rect :x="x(i)" y="179" width="36" height="32" class="gene" />
        <text :x="x(i) + 18" y="198" class="bit">{{ col.parent === 'A' ? col.a : col.b }}</text>
      </g>
    </svg>
    <figcaption>Illustrative feature masks. A <code>1</code> keeps that column; a <code>0</code> drops it.</figcaption>
  </figure>
</template>

<style scoped>
.crossover {
  --diagram-ink: #272c28;
  --diagram-muted: #515852;
  --diagram-accent: #a53320;
  --diagram-wash: #f3e4e0;
  --diagram-rule: #9b9d95;
  margin: 0;
  min-width: 0;
  color: var(--diagram-ink);
  padding-block: 0.75rem 0.85rem;
  border-block: 1px solid var(--diagram-rule);
}
.dark .crossover {
  --diagram-ink: #eceee8;
  --diagram-muted: #bec5ba;
  --diagram-accent: #ff9a7e;
  --diagram-wash: #432b25;
  --diagram-rule: #777b72;
}
.crossover-heading {
  margin: 0 0 0.7rem;
  font-size: clamp(1rem, 2.2vw, 1.25rem);
  font-weight: 650;
  line-height: 1.4;
}
.crossover-diagram {
  display: block;
  width: 100%;
  height: auto;
  overflow: visible;
}
.row-label,
.column-label,
.parent-label,
.bit {
  dominant-baseline: middle;
  fill: var(--diagram-ink);
}
.row-label {
  font-size: 15.5px;
}
.column-label,
.parent-label,
.bit {
  text-anchor: middle;
  font-family: var(--vp-font-family-mono);
}
.column-label {
  font-size: 14px;
  fill: var(--diagram-muted);
}
.parent-label,
.bit {
  font-size: 17px;
}
.inherited .parent-label,
.inherited .bit,
.offspring-label {
  font-weight: 700;
}
.gene {
  fill: none;
  stroke: var(--diagram-rule);
}
.inherited .gene {
  stroke: var(--diagram-ink);
  stroke-width: 1.6;
}
.inherited.from-b .gene {
  fill: var(--diagram-wash);
  stroke: var(--diagram-accent);
}
.inherited.from-b :is(.bit, .parent-label) {
  fill: var(--diagram-accent);
}
.row-rule,
.inheritance-line {
  fill: none;
  stroke: var(--diagram-rule);
}
.inheritance-line {
  stroke: var(--diagram-ink);
  stroke-width: 1.4;
}
.from-b .inheritance-line {
  stroke: var(--diagram-accent);
}
.crossover figcaption {
  margin-top: 0.4rem;
  color: var(--diagram-muted);
  font-size: 0.875rem;
  line-height: 1.5;
}
</style>
