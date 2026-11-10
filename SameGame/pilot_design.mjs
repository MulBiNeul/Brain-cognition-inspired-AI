import { readFileSync } from 'node:fs';

const baseline = JSON.parse(readFileSync(new URL('./pilot_v3_baseline.json', import.meta.url), 'utf8')).items;

// A private stream keeps the study design independent of tape-generation retries.
function seededShuffle(seed) {
  return values => {
    const result = [...values];
    for (let i = result.length - 1; i > 0; i--) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      const j = Math.floor(seed / 2 ** 32 * (i + 1));
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  };
}

const goals = [0, 1, 2, 3];
const edgeIds = row => row.slice(1).map((next, i) => 4 * row[i] + next);
const repeatedGoal = row => row.find((goal, i) => row.indexOf(goal) !== i);

function goalSchedules(shuffle) {
  let all = [[]];
  for (let i = 0; i < 5; i++) {
    all = all.flatMap(row => goals.filter(goal => goal !== row.at(-1)).map(goal => [...row, goal]));
  }
  all = all.filter(row => new Set(row).size === 4);
  const byRepeatPosition = new Map();
  for (const row of all.filter(row => row[0] === 0)) {
    const g = repeatedGoal(row);
    const key = row.indexOf(g) + ',' + row.lastIndexOf(g);
    if (!byRepeatPosition.has(key)) byRepeatPosition.set(key, []);
    byRepeatPosition.get(key).push(row);
  }

  // Each four-way relabelling contributes one occurrence of each goal at
  // every round. Twelve groups cover 48 items; two further rows finish 50.
  for (let attempt = 0; attempt < 10000; attempt++) {
    const representatives = [...byRepeatPosition.values()].flatMap(rows => shuffle(rows).slice(0, 2));
    const rows = representatives.flatMap(row => goals.map(offset => row.map(g => (g + offset) % 4)));
    const edges = rows.flatMap(edgeIds);
    const distinct = new Set(edges);
    if (distinct.size !== 12 || [...distinct].some(edge => edges.filter(x => x === edge).length !== 16)) continue;
    const used = new Set(rows.map(r => r.join('')));
    const extra = shuffle(all.filter(r => !used.has(r.join('')) && new Set(edgeIds(r)).size === 4));
    for (const a of extra) {
      const b = extra.find(b => a.every((goal, i) => goal !== b[i]) &&
        repeatedGoal(a) !== repeatedGoal(b) && new Set([...edgeIds(a), ...edgeIds(b)]).size === 8 &&
        (a.indexOf(repeatedGoal(a)) !== b.indexOf(repeatedGoal(b)) ||
         a.lastIndexOf(repeatedGoal(a)) !== b.lastIndexOf(repeatedGoal(b))));
      if (b) return [...rows, a, b];
    }
  }
  throw Error('Could not balance the directed goal transitions.');
}

function sideSchedules(shuffle) {
  const switches = baseline.map(item => item.priorities.findIndex(p => p !== item.priorities[0]));
  const result = Array(baseline.length);
  const turnCounts = {2: [[1, 6], [3, 12], [4, 7]], 3: [[1, 7], [2, 12], [4, 6]]};
  for (const priorityTurn of [2, 3]) {
    const indices = shuffle(baseline.map((_, i) => i).filter(i => switches[i] === priorityTurn));
    const sideTurns = turnCounts[priorityTurn].flatMap(([turn, count]) => Array(count).fill(turn));
    if (indices.length !== sideTurns.length) throw Error('Unexpected v3 priority schedule.');
    indices.forEach((i, j) => { result[i] = {priorityTurn, sideTurn: sideTurns[j]}; });
  }
  for (const turn of [1, 2, 3, 4]) {
    const indices = shuffle(baseline.map((_, i) => i).filter(i => result[i].sideTurn === turn));
    const leftCount = turn === 1 ? 7 : 6;
    const example = indices.findIndex(i => baseline[i].id === 'SG-036');
    if (example >= 0) {
      const slot = turn === 1 ? leftCount : 0;
      [indices[example], indices[slot]] = [indices[slot], indices[example]];
    }
    indices.forEach((i, j) => {
      result[i].sides = Array.from({length: 5}, (_, r) => (j < leftCount) === (r < turn) ? 'left' : 'right');
    });
  }
  return result;
}

function reuseGoalRows(rows, configs, example, makeSpec, canReuse) {
  const groups = new Map();
  configs.forEach(({priorityTurn, sideTurn}, i) => {
    if (i === example) return;
    const key = priorityTurn + ',' + sideTurn;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(i);
  });
  const assigned = [...rows];
  for (const indices of groups.values()) {
    // The largest group has 12 items. Assign every row exactly once,
    // maximizing reusable tapes first and unchanged row assignments second.
    const n = indices.length, full = (1 << n) - 1;
    const weights = indices.map(i => indices.map(j =>
      Number(canReuse(makeSpec(i, rows[j]))) * (n + 1) + Number(i === j)));
    const scores = new Map([[full, 0]]), picks = new Map();
    function best(mask, i) {
      if (scores.has(mask)) return scores.get(mask);
      let highest = -1;
      for (let j = 0; j < n; j++) {
        const bit = 1 << j;
        if (mask & bit) continue;
        const score = weights[i][j] + best(mask | bit, i + 1);
        if (score > highest) { highest = score; picks.set(mask, j); }
      }
      scores.set(mask, highest);
      return highest;
    }
    best(0, 0);
    let mask = 0;
    for (const i of indices) {
      const j = picks.get(mask);
      assigned[i] = rows[indices[j]];
      mask |= 1 << j;
    }
  }
  return assigned;
}

export function studyDesign(canReuse) {
  const shuffle = seededShuffle(2026111004);
  const rows = goalSchedules(shuffle);
  const configs = sideSchedules(shuffle);
  const example = baseline.findIndex(item => item.id === 'SG-036');
  let best, lowest = Infinity;
  // Spread each old/new target combination across the three kinds of transition.
  for (let attempt = 0; attempt < 800; attempt++) {
    const proposal = shuffle(rows);
    const other = proposal.findIndex(row => row[1] === 3);
    [proposal[example], proposal[other]] = [proposal[other], proposal[example]];
    const counts = Array.from({length: 3}, () => Array(16).fill(0));
    proposal.forEach((row, i) => edgeIds(row).forEach((edge, j) => {
      const kind = configs[i].priorityTurn === j + 1 ? 1 : configs[i].sideTurn === j + 1 ? 2 : 0;
      counts[kind][edge]++;
    }));
    const score = counts.flat().reduce((sum, n) => sum + n * n, 0);
    if (score < lowest) { best = proposal; lowest = score; }
  }
  function makeSpec(i, row) {
    const item = baseline[i];
    return {
      id: item.id, input: item.input, spaced: item.input.includes(' '), exactExample: i === example,
      v3Goals: [...item.v3Goals], v3Rules: [...item.v3Rules],
      settings: row.map((target_length, r) => ({
        round: r + 1, priority: item.priorities[r], deletion_rule: 'whole',
        selection_side: configs[i].sides[r], target_length,
      })),
    };
  }
  const selected = canReuse ? reuseGoalRows(best, configs, example, makeSpec, canReuse) : best;
  return selected.map((row, i) => makeSpec(i, row));
}
