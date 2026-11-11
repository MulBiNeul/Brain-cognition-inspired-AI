import { writeFileSync } from 'node:fs';
import { analyze, uniformPlayOutcomes } from './solve.mjs';

const seed = 2026111101;
let state = seed;
const alphabet = 'ABCDE';

function randomInt(limit) {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
  return Math.floor(state / 2 ** 32 * limit);
}
function pick(values) {
  return values[randomInt(values.length)];
}
function shuffled(values) {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i--) {
    const j = randomInt(i + 1);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
function goalTemplate() {
  const row = shuffled([0, 1, 2, 3]);
  const repeat = pick(row);
  const positions = [0, 1, 2, 3, 4].filter(i => row[i] !== repeat && row[i - 1] !== repeat);
  row.splice(pick(positions), 0, repeat);
  return row;
}
function goalsForFamily(extra) {
  const rows = [];
  for (let i = 0; i < 6; i++) {
    const base = goalTemplate();
    for (let k = 0; k < 4; k++) rows.push(base.map(x => (x + k) % 4));
  }
  return shuffled([...rows, extra]);
}
function shiftPair(pair, n) {
  return [...pair].map(c => alphabet[(alphabet.indexOf(c) + n) % 5]).join('');
}
function rulesForFamily(type) {
  const rows = [];
  const offsets = shuffled([1, 2, 3, 4]);
  for (let group = 0; group < 5; group++) {
    let base;
    if (type === 'same_letter') {
      base = shuffled(alphabet).map(c => c + c);
    } else {
      // Keep one endpoint at each switch. Cycling the offsets across groups
      // balances all 20 ordered pairs without treating AC and CA as equivalent.
      const distances = Array.from({length: 5}, (_, i) => offsets[(i + group) % 4]);
      do {
        let start = 0, end = distances[0];
        base = [alphabet[start] + alphabet[end]];
        for (const d of distances.slice(1)) {
          if (randomInt(2)) start = (end - d + 5) % 5;
          else end = (start + d) % 5;
          base.push(alphabet[start] + alphabet[end]);
        }
      } while (new Set(base).size !== 5);
    }
    for (let k = 0; k < 5; k++) rows.push(base.map(pair => shiftPair(pair, k)));
  }
  return shuffled(rows);
}
function design() {
  const extra = goalTemplate();
  return ['different_letter', 'same_letter'].flatMap((type, f) => {
    const goals = goalsForFamily(extra.map(x => (x + f) % 4));
    const spaces = shuffled(Array.from({length: 25}, (_, i) => i < 10));
    return rulesForFamily(type).map((rules, i) => ({
      id: 'SG-' + String(25 * f + i + 1).padStart(3, '0'), type, rules,
      targets: goals[i], spaced: spaces[i],
    }));
  });
}

// Reverse-insert removable groups. Inserting inside an equal pair creates
// isolated letters which must later be reunited by deleting an interior run.
function makeTape(size, target, spaced, pair, exact, wantThree) {
  const parts = exact === null ? Array(spaced ? 2 : 1).fill('') : exact.split(' ');
  const others = shuffled([...alphabet].filter(c => !pair.includes(c)));
  if (exact === null) {
    for (let i = 0; i < target; i++) {
      const j = randomInt(parts.length);
      const pool = (target > 1 && i > 0 && randomInt(4) ? others : [...alphabet])
        .filter(c => !parts[j].endsWith(c));
      parts[j] += pick(pool);
    }
  }
  // XYYXXY can reduce to X or Y. The longer variants also have
  // completions of different lengths. Expand one surviving letter into
  // such a fork before wrapping it in the ordinary insertion scaffold.
  if (target > 0 && randomInt(12) !== 0) {
    const j = pick(parts.map((s, j) => s ? j : -1).filter(j => j >= 0));
    const at = randomInt(parts[j].length);
    const x = parts[j][at];
    const y = pick([...new Set([...pair, ...others])].filter(c => c !== x));
    const core = pick(['XYYXXY', 'XYYXXY', 'XYYXXYXX', 'XYYXXYXX', 'XYXXYXYYXXY']);
    let fork = [...core].map(c => c === 'X' ? x : y).join('');
    if (wantThree && randomInt(3) !== 0) {
      // Removing different pairs first leaves X, Y or Z in each small core.
      const pattern = pick(['XZZXXZYYXXY', 'XYYXXYZZXXZ', 'ZXYYXZZXYYXXY', 'XYYZZZXXZYXXY']);
      const a = pair[0];
      let b = pair[1];
      if (a === b) b = y === a ? pick(others) : y;
      const c = x !== a && x !== b ? x : pick([...alphabet].filter(color => color !== a && color !== b));
      const colors = {X: a, Y: b, Z: c};
      fork = [...pattern].map(symbol => colors[symbol]).join('');
    }
    parts[j] = parts[j].slice(0, at) + fork + parts[j].slice(at + 1);
  }
  const widths = [];
  for (let left = size - parts.join('').length; left > 0;) {
    const w = left === 3 || (left >= 5 && randomInt(12) === 0) ? 3 : 2;
    widths.push(w);
    left -= w;
  }
  // In some candidates reconstruct a witness backwards, leaving places to
  // interrupt the active pair. The solver independently tests both settings.
  const focus = randomInt(4) !== 0;
  const slots = widths.length;
  let order = [];
  if (focus && pair[0] === pair[1]) {
    const repeats = Math.floor(slots / 2) + randomInt(3) - 1;
    const separators = [...others];
    while (separators.length < slots - repeats) separators.push(pick(others));
    order = [...shuffled(separators), ...Array(repeats).fill(pair[0])];
  } else if (focus) {
    const fillers = shuffled([...alphabet].filter(c => !pair.includes(c) && !parts.join('').includes(c)));
    const repeats = Math.floor((slots - fillers.length) / 2);
    order = [...pair.repeat(repeats)];
    while (order.length + fillers.length < slots) fillers.push(pick(alphabet));
    for (const c of fillers) {
      // Insert between complete occurrences except at the ends.
      const k = randomInt(Math.floor(order.length / 2) + 1) * 2;
      order.splice(k, 0, c);
    }
    order.reverse();
  }
  const first = spaced ? 3 + randomInt(widths.length - 5) : widths.length;
  const segments = shuffled(widths.map((_, k) => k < first ? 0 : 1));
  const pool = [...others, ...pair.repeat(2 + randomInt(5))];
  for (const [i, w] of widths.entries()) {
    const j = segments[i], s = parts[j];
    const joins = [], allowed = [], planned = order[i];
    for (let k = 0; k <= s.length; k++) {
      if (planned && (s[k] === planned || s[k - 1] === planned)) continue;
      allowed.push(k);
      if (k > 0 && k < s.length && s[k] === s[k - 1]) joins.push(k);
    }
    if (!allowed.length) return null;
    const at = joins.length && randomInt(2) === 0 ? pick(joins) : pick(allowed);
    const letter = planned ?? pick(pool.filter(c => c !== s[at - 1] && c !== s[at]));
    parts[j] = s.slice(0, at) + letter.repeat(w) + s.slice(at);
  }
  return parts.join(' ');
}
function canonical(input) {
  const names = new Map();
  return Array.from(input, c => {
    if (c === ' ') return c;
    if (!names.has(c)) names.set(c, alphabet[names.size]);
    return names.get(c);
  }).join('');
}
function assess(input, row, minimumMultisets = 0, preferFirst = false) {
  const {spec, i, exact} = row;
  const forbidden = spec.rules[i], target = spec.targets[i];
  const previous = i ? spec.rules[i - 1] : null;
  if (!input || ![...alphabet].every(c => input.includes(c))) return null;
  if (new Set(Array.from(input.matchAll(/([A-E])\1+/g), m => m[1])).size < 2) return null;
  // A synthetic comparison rule makes R1 diagnostic too. It is never scored as a previous rule.
  const stats = analyze(input, target, forbidden, previous ?? spec.rules[1], exact, previous !== null,
    minimumMultisets, spec.spaced, preferFirst);
  if (!stats) return null;
  const probe = stats.previous_rule_probe;
  const recoverable = probe.recoverable_violation_steps.length;
  if (recoverable < (previous ? 3 : 2) || (previous && 5 * recoverable < 2 * probe.steps)) return null;
  const odds = uniformPlayOutcomes(input, target, forbidden);
  const share = target === 0 ? odds.success : odds.mostCommonGivenSuccess;
  if (share > 0.95 + 1e-12 || odds.successfulMoves < 3) return null;
  return {entry: {round: i + 1, input, target, forbidden},
    key: {round: i + 1, answer: stats.final_state}, odds};
}

const plan = design();
// Every goal receives the same candidate lengths within its family and round.
// Different-letter transition probes with target one need at least 31 letters.
const sizes = plan.map(() => Array(5));
function assignSizes(type, rounds, options) {
  const offset = [0, 1, 2, 3].map(target => Math.floor(target / 2) % options.length);
  for (const round of rounds) {
    for (let target = 0; target <= 3; target++) {
      const items = [];
      plan.forEach((spec, i) => {
        if (spec.type === type && spec.targets[round] === target) items.push(i);
      });
      for (const item of shuffled(items)) {
        sizes[item][round] = options[offset[target]];
        offset[target] = (offset[target] + 1) % options.length;
      }
    }
  }
}
const allSizes = [28, 29, 30, 31, 32];
assignSizes('different_letter', [0], allSizes);
assignSizes('different_letter', [1, 2, 3, 4], [31, 32]);
assignSizes('same_letter', [0, 1, 2, 3, 4], allSizes);
// The previous release's per-round PRNG states reproduce its input tapes
// without storing or requiring the old dataset. Revalidate each tape under
// the current rules before generating replacements for rejected rounds.
const initialSeeds = [
  [1845544896, 3451652598, 553180424, 3813764372, 3145648296],
  [128785280, 3338060521, 3384823361, 2694485291, 4235993284],
  [1371706483, 1869800657, 3423506811, 3016249905, 1338824070],
  [1371691410, 3939512782, 2707048895, 3487534540, 861803111],
  [2481076807, 3382381879, 4051763747, 346526140, 222127566],
  [40643920, 2549766001, 3219898759, 3319060597, 3279061628],
  [4278206443, 2545641979, 2933995474, 3493131400, 3280793096],
  [221445667, 1880947828, 1579757139, 2880081401, 3910963079],
  [4185807772, 1687868177, 2971519553, 4245008131, 2279229044],
  [4113656337, 1223164874, 547578516, 618979428, 2371216691],
  [2656961630, 3863104965, 2772959047, 3626445371, 1244628393],
  [4264348270, 305403972, 3658355524, 719168254, 755067352],
  [1095077754, 3435676259, 1849087963, 4154563591, 501351115],
  [3513660220, 2503021607, 2171262303, 1524031043, 2685290314],
  [2332496956, 345565624, 3363954178, 3879225196, 2799645040],
  [545758843, 2132327349, 2590305774, 3501421534, 4111130902],
  [1649234429, 560415480, 1876878861, 2336286043, 2527087545],
  [2813980035, 1804790221, 3350239578, 2588929869, 1793740328],
  [4020546386, 2668736431, 1698483381, 565471298, 2648547756],
  [1991977292, 3336617885, 1595645509, 3416428018, 476641666],
  [1395172927, 3258113563, 3655635343, 1522324458, 4187643177],
  [3626531943, 450836290, 2612936903, 1288746707, 439155032],
  [41337828, 4024220176, 1860732898, 1290045946, 3989503884],
  [2770848187, 3607727648, 3250099911, 2934704683, 1180082433],
  [324794271, 836700296, 1901921582, 2087805045, 1270794816],
  [3948348833, 1065186803, 1228742920, 4208742800, 2133186126],
  [123180299, 1608840295, 1335072007, 1179036550, 2598690197],
  [3353387485, 629657944, 1120581851, 908384872, 4217424748],
  [790068157, 843298641, 3455628929, 4284395441, 781625708],
  [4212715140, 3613501538, 2566548571, 1397838941, 443316228],
  [755334499, 985227157, 2090023093, 2735649021, 2383420559],
  [3394799307, 3869262955, 3014068110, 1976017889, 237039830],
  [83831101, 2242283846, 352021427, 1653871534, 4157112033],
  [1521042835, 1012681580, 3472998125, 1133090179, 3052900073],
  [1055457559, 3875273640, 1187818591, 1638335186, 2025230377],
  [2118593250, 1648289446, 534680701, 1351370727, 1613546706],
  [3589921608, 1679419907, 2938993956, 1348909572, 1851557531],
  [439729028, 1589134585, 2137283387, 2588133905, 852321211],
  [2810538383, 2975843378, 3469266567, 1406098275, 2072257149],
  [929152181, 2817132561, 4278766020, 1698882841, 3072107215],
  [174095441, 2841372399, 2593623625, 3310752180, 307749547],
  [1016381400, 3669973891, 2038295429, 453912518, 1042241746],
  [3153369917, 2895209694, 1443475804, 3784080469, 4000736148],
  [134287605, 1732313993, 1554718411, 3640660888, 3701053343],
  [1954457027, 626790653, 3852360944, 2164231885, 3121910360],
  [844022943, 3686862905, 289516061, 3730072260, 257351991],
  [2550656805, 3989540069, 710034115, 68910335, 2563796485],
  [459482746, 3774738100, 2563340838, 1729140734, 1392051427],
  [569168784, 2453775209, 1054689914, 3288535544, 1600286172],
  [1988840029, 103463867, 1085391749, 3676227231, 2548010204],
];
const expandedCores = new Set([0, 1, 3, 19, 39, 40, 44, 53, 60, 70, 90, 97, 102, 105, 152, 160, 184, 185, 191, 192, 200, 205, 210, 237, 239, 243, 246]);
const seen = new Set(), boards = [];
const boundaryExamples = new Map([[2, 'A A'], [3, 'AB A']]);
for (const [index, spec] of plan.entries()) {
  for (const [i, target] of spec.targets.entries()) {
    const exact = spec.spaced && i === 0 ? boundaryExamples.get(target) ?? null : null;
    const row = {spec, i, exact, size: sizes[index][i]};
    state = initialSeeds[index][i];
    const input = makeTape(row.size, target, spec.spaced, spec.rules[i], exact, expandedCores.has(boards.length));
    row.result = assess(input, row, 0, true);
    if (row.result) seen.add(canonical(input));
    if (exact !== null) boundaryExamples.delete(target);
    boards.push(row);
  }
}

function generateReplacement(row, minimumMultisets) {
  const {spec, i, size, exact} = row;
  const target = spec.targets[i], forbidden = spec.rules[i];
  for (let attempt = 0; attempt < 100000; attempt++) {
    const input = makeTape(size, target, spec.spaced, forbidden, exact, minimumMultisets >= 3);
    if (!input || seen.has(canonical(input))) continue;
    const result = assess(input, row, minimumMultisets);
    if (result) {
      seen.add(canonical(input));
      return result;
    }
  }
  throw Error('No qualifying round for ' + spec.id + ' R' + (i + 1));
}

state = 2026111181;
for (const target of [0, 1, 2, 3]) {
  const group = boards.filter(row => row.spec.targets[row.i] === target);
  let rich = group.filter(row => row.result?.odds.multisetCount >= 3).length;
  const required = target ? Math.ceil(group.length / 4) : 0;
  // Fill the diversity quota from replacement slots first, preferring R1
  // where generating three different final compositions is less constrained.
  const missing = group.filter(row => !row.result).sort((a, b) => a.i - b.i);
  for (const row of missing) {
    row.result = generateReplacement(row, rich < required ? 3 : 0);
    if (row.result.odds.multisetCount >= 3) rich++;
  }
  if (rich < required) throw Error('Insufficient distinct final multisets for target ' + target);
}

const items = [], answers = [];
for (const spec of plan) {
  const rounds = [], keys = [];
  for (const row of boards.filter(row => row.spec === spec)) {
    rounds.push(row.result.entry);
    keys.push(row.result.key);
  }
  items.push({id: spec.id, type: spec.type, rounds});
  answers.push({id: spec.id, rounds: keys});
}
const rules_en = [
  'Each item has five consecutive rounds. Each round has one input. Every submission in that round starts from the same input with an empty action log.',
  'A move names a letter A through E and the ordinal of one of its current maximal runs of two or more identical letters. Number only that letter\'s removable runs from left to right, starting at 1, across the entire string. On AABAA, A2 deletes the right-hand AA. Remove the selected whole run, join its neighbors within that segment, and renumber runs for the next move. A newly joined run remains until selected.',
  'A literal ASCII space is a permanent boundary. Letters cannot join across it. Ignore spaces when counting letters, but keep the boundary when checking adjacency.',
  'The two letters in forbidden specify one prohibited pair of consecutive deletion letters. Ignore the run ordinals when checking this pair. AC prohibits C immediately after A, but permits A after C. BB prohibits selecting B twice in succession. Deleting BBB once is one action. The first action is unrestricted by this rule.',
  'Replay the submitted commands in order. Continue deleting even after reaching or passing the target length. Record each forbidden pair and execute the deletion if it is physically possible. If the named run does not exist, check the attempted letter pair, stop at that command and mark the plan invalid.',
  'Judge the final board after replay. For success, all submitted commands must execute, the action log must obey the forbidden pair, exactly target letters must remain, and no removable run may remain in any segment. At target 2, "A A" succeeds and AA fails. At target 3, ABA, "AB A" and "A AB" succeed; AAB, ABB and "AA B" fail.',
  'The evaluator keeps the exact target and forbidden pair in private round metadata. Submit one JSON object {"moves":"...","answer":"..."}: moves is a string of 1 to 30 commands such as "A2 B1 C1", separated by one ASCII space. Use positive decimal ordinals without leading zeros. A plain letter string such as "ABC" selects ordinal 1 on every move. The answer is the final string produced by those moves. Keep internal spaces, omit leading and trailing spaces, and use the empty string when no letters remain.',
].join(' ');
for (const [file, content] of [
  ['problems.json', {rules_en, rounds_per_item: 5, items}],
  ['answer_key.json', {items: answers}],
]) writeFileSync(new URL(`../data/${file}`, import.meta.url), JSON.stringify(content, null, 2) + '\n');
console.log('Generated ' + items.length + ' items and ' + seen.size + ' distinct round inputs.');
