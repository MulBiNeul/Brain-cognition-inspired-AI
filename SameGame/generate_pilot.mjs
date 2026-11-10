import { writeFileSync } from 'node:fs';

// Run with Node.js. Both JSON files are written beside this script.
const alphabet = 'ABCDE';
const seed = 20261110;
let randomState = seed;
function randomInt(limit) {
  randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
  return Math.floor((randomState / 2 ** 32) * limit);
}
function shuffle(values) {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i--) {
    const j = randomInt(i + 1);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
const lettersOnly = state => state.replaceAll(' ', '');
const length = state => lettersOnly(state).length;

function studyDesign() {
  const orders = [];
  // Across 10 rows, each letter occupies each position twice and every
  // ordered pair of distinct priorities occurs twice.
  for (let copy = 0; copy < 5; copy++) {
    for (const offsets of [[0, 1, 4, 2, 3], [3, 2, 4, 1, 0]]) {
      for (let shift = 0; shift < 5; shift++) {
        orders.push(offsets.map(i => alphabet[(shift + i) % 5]).join(''));
      }
    }
  }
  const shuffledOrders = shuffle(orders);
  const design = Array.from({length: 50}, (_, i) => ({
    id: 'SG-' + String(i + 1).padStart(3, '0'),
    target: i % 4, order: shuffledOrders[i], spaced: false, exactExample: false,
  }));
  for (const target of [0, 1, 2, 3]) {
    const spaced = shuffle(design.filter(item => item.target === target)).slice(0, 5);
    for (const item of spaced) item.spaced = true;
    if (target === 3) spaced[0].exactExample = true;
  }
  return design;
}

function makeInput(spec) {
  const tokenCount = 28 + randomInt(5);
  const split = spec.spaced ? 8 + randomInt(tokenCount - 15) : tokenCount;
  function segment(size) {
    let state = '';
    while (state.length < size) {
      const choices = [...alphabet].filter(c => c !== state.at(-1));
      const letter = choices[randomInt(choices.length)];
      const draw = randomInt(100);
      const repeats = draw < 19 ? 1 : draw < 93 ? 2 : 3;
      state += letter.repeat(Math.min(repeats, size - state.length));
    }
    return state;
  }
  return spec.spaced ? segment(split) + ' ' + segment(tokenCount - split) : segment(split);
}

function engine() {
  const moveCache = new Map();
  function moves(state) {
    if (moveCache.has(state)) return moveCache.get(state);
    const result = new Map();
    for (const match of state.matchAll(/([A-E])\1+/g)) {
      const letter = match[1];
      if (!result.has(letter)) {
        result.set(letter, state.slice(0, match.index) + state.slice(match.index + match[0].length));
      }
    }
    moveCache.set(state, result);
    return result;
  }
  function allowed(state, priority) {
    const options = moves(state);
    return options.has(priority) ? [priority] : [...options.keys()];
  }
  return {moves, allowed};
}

// The memoized search counts every valid action sequence, including alternate
// solutions that leave different letters. The exact-terminal search is used
// only to exhibit one answer ending in "AB A", never as an acceptance rule.
function solve(start, spec, priority, nextPriority, game, exhibit = false) {
  const memo = new Map();
  function visit(state) {
    if (memo.has(state)) return memo.get(state);
    const size = length(state);
    if (size === spec.target) {
      return !exhibit || state === 'AB A' ? {count: 1, min: 0, max: 0, path: ''} : null;
    }
    if (size < spec.target + 2) return null;
    const options = game.moves(state);
    let result = null;
    for (const action of game.allowed(state, priority)) {
      const child = visit(options.get(action));
      if (!child) continue;
      const hit = Number(action === priority && options.has(nextPriority));
      if (!result) result = {count: 0, min: Infinity, max: -1, path: ''};
      result.count += child.count;
      result.min = Math.min(result.min, child.min + hit);
      if (child.max + hit > result.max) {
        result.max = child.max + hit;
        result.path = action + child.path;
      }
    }
    memo.set(state, result);
    return result;
  }
  return visit(start);
}

function replay(start, path, old, next, game) {
  let state = start;
  const hits = [];
  for (const [i, action] of [...path].entries()) {
    const options = game.moves(state);
    if (action === old && options.has(next)) hits.push(i + 1);
    state = options.get(action);
  }
  return {final: state, hits};
}

function spaceProbe(start, target, round, game) {
  let spaced = start, joined = lettersOnly(start);
  let first = null;
  const base = {round: round.round, priority: round.priority, one_valid_answer: round.one_valid_answer};
  for (const [i, action] of [...round.one_valid_answer].entries()) {
    const step = i + 1;
    let reason;
    if (length(joined) === target) reason = 'continued_after_goal';
    else if (!game.moves(joined).has(action)) reason = 'no_removable_run';
    else if (!game.allowed(joined, round.priority).includes(action)) reason = 'priority_violation';
    if (reason) {
      return {...base, first_different_step: first ?? step, failure_step_without_space: step,
        failure_without_space: reason, state_without_space_at_failure: joined};
    }
    spaced = game.moves(spaced).get(action);
    joined = game.moves(joined).get(action);
    if (first === null && lettersOnly(spaced) !== joined) first = step;
  }
  if (length(joined) === target) return null;
  return {...base, first_different_step: first, failure_step_without_space: round.one_valid_answer.length,
    failure_without_space: 'wrong_final_length', state_without_space_at_failure: joined};
}

function evaluate(input, spec) {
  const segments = input.split(' ');
  // The two facing letters start out different; the effect of a boundary
  // arises during deletion, rather than splitting an already visible pair.
  if (spec.spaced && segments[0].at(-1) === segments[1][0]) return null;
  if (spec.exactExample && (!segments[0].includes('A') || !segments[0].includes('B') || !segments[1].includes('A'))) return null;
  for (const letter of spec.order.slice(0, 4)) {
    if (input.split(letter).length - 1 < 4) return null;
  }
  const game = engine(), rounds = [], transitions = [];
  let exhibited = false, late = false;
  for (let i = 0; i < 5; i++) {
    const p = spec.order[i], next = spec.order[i + 1];
    const result = solve(input, spec, p, next, game);
    if (!result || (next && result.min < 2)) return null;
    let path = result.path;
    if (spec.exactExample && !exhibited) {
      const witness = solve(input, spec, p, next, game, true);
      if (witness) { path = witness.path; exhibited = true; }
    }
    const trace = replay(input, path, p, next, game);
    rounds.push({round: i + 1, priority: p, one_valid_answer: path, final_state: trace.final,
      n_valid_answers: result.count});
    if (next) {
      late ||= trace.hits.some((step, j, hits) => j > 0 && step >= 5 && step > hits[j - 1] + 1);
      transitions.push({from_round: i + 1, to_round: i + 2, old_priority: p, new_priority: next,
        old_priority_matches_on_replayed_reference: trace.hits,
        min_over_all_previous_solutions: result.min, max_over_all_previous_solutions: result.max});
    }
  }
  if (!late || (spec.exactExample && !exhibited)) return null;
  const probe = spec.spaced ? rounds.map(r => spaceProbe(input, spec.target, r, game)).find(Boolean) : null;
  if (spec.spaced && !probe) return null;
  return {id: spec.id, rounds, transitions, space_probe: probe};
}

const inputItems = [], answerItems = [], seen = new Set();
for (const spec of studyDesign()) {
  let found = false;
  for (let attempt = 0; attempt < 250000 && !found; attempt++) {
    const input = makeInput(spec);
    if (seen.has(input)) continue;
    const answer = evaluate(input, spec);
    if (!answer) continue;
    inputItems.push({id: spec.id, input, target_length: spec.target});
    answerItems.push(answer);
    seen.add(input);
    found = true;
  }
  if (!found) throw Error('No qualifying input for ' + spec.id);
}
const version = 'pilot-2';
const problems = {
  version, rounds_per_item: 5,
  rules_en: 'Start from a string over A through E. An action names one letter and deletes its leftmost removable run in full. A removable run is two or more consecutive identical letters in the same segment. After deletion, the letters on either side become neighbors within that segment. If a run of the current priority letter is removable, you must choose it; otherwise you may choose any removable letter.',
  space_rule_en: 'Some inputs contain one literal ASCII space. It is a permanent boundary: letters on opposite sides never form a run. Inputs without a space are a single segment. Ignore spaces only to count the letters remaining.',
  space_example_en: 'With priority B and a target of zero, "AABBAA" clears in two moves: BA. With "AABB AA", deleting B leaves "AA AA". The two A runs stay separate, so clearing it takes three moves: BAA.',
  goal_en: 'Stop as soon as exactly target_length letters remain. Their identities and order are unrestricted. For target_length 0, remove all letters; a remaining space still counts as zero. For example, "AB A" has length 3 and "AA" is already a success at target_length 2 even though it is removable.',
  reset: 'Restart from the same original input and target_length in each round. Only the priority changes. Return the letters you delete, in order.',
  items: inputItems,
};
const keys = {version, seed,
  evaluator_notes: 'An answer is an action sequence, not the terminal string. Replay on the original input. Require physically removable actions, adherence to priority, and stopping on the first exact target length. Accept every valid terminal string. Character offsets in input are not action positions. All step numbers below are one-based. For carryover diagnostics on physical replay, record a previous-priority match when both priorities are removable and the old letter is chosen. Keep replaying priority violations if physically legal; stop at the goal or a physically illegal action. Each space_probe gives a valid bounded answer that fails if the space is erased under the same priority and target. first_different_step marks the first action with a different resulting letter string or an invalid submission in the boundary-free replay.',
  items: answerItems,
};
for (const [name, content] of [['problems.json', problems], ['answer_key.json', keys]]) {
  writeFileSync(new URL(name, import.meta.url), JSON.stringify(content, null, 2) + '\n');
}
console.log('Generated ' + inputItems.length + ' pilot items.');
