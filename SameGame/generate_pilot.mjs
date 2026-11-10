import { writeFileSync } from 'node:fs';

const alphabet = 'ABCDE';
const seed = 2026111003;
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
function terminal(state, target) {
  const letters = lettersOnly(state);
  if (letters.length !== target) return null;
  return target === 2 && letters[0] === letters[1] ? 'duplicate_final_pair' : 'goal';
}

function studyDesign() {
  const pairs = [];
  // Reverse five complete shift groups: each direction then contains all
  // five old and all five new priorities equally often at the middle round.
  const directions = shuffle(Array.from({length: 10}, (_, i) => i < 5));
  for (const offset of [1, 2, 3, 4, 1, 2, 3, 4, 1, 3]) {
    const reverse = directions.pop();
    for (let i = 0; i < 5; i++) pairs.push([alphabet[i], alphabet[(i + offset) % 5], reverse]);
  }
  const orders = shuffle(pairs);
  const design = orders.map(([oldPriority, newPriority, reverse], i) => {
    const t = i % 4, u = (t + 1) % 4;
    const [p, q] = reverse ? [newPriority, oldPriority] : [oldPriority, newPriority];
    const settings = [['whole', t, p], ['pair', t, p], ['pair', t, q], ['whole', t, q], ['whole', u, q]];
    if (reverse) settings.reverse();
    return {id: 'SG-' + String(i + 1).padStart(3, '0'), target: t, spaced: false, exactExample: false,
      settings: settings.map(([deletion_rule, target_length, priority], j) =>
        ({round: j + 1, deletion_rule, target_length, priority}))};
  });
  for (const target of [0, 1, 2, 3]) {
    const spaced = shuffle(design.filter(item => item.target === target)).slice(0, 5);
    for (const item of spaced) item.spaced = true;
    if (target === 3) spaced[0].exactExample = true;
  }
  return design;
}

// Inserting identical pairs preserves letter parity in each segment, making
// the pair rule feasible even for zero-letter goals on tapes with boundaries.
function makeInput(spec) {
  const target = spec.target;
  const sizes = target % 2 ? [29, 31] : [28, 30, 32];
  const size = sizes[randomInt(sizes.length)];
  const parts = spec.spaced ? ['', ''] : [''];
  if (spec.exactExample) {
    parts[0] = 'AB'; parts[1] = 'A';
  } else {
    const finalLetters = shuffle(alphabet).slice(0, target);
    for (const c of finalLetters) parts[randomInt(parts.length)] += c;
  }
  const pairCount = (size - target) / 2;
  const firstCount = spec.spaced ? 4 + randomInt(pairCount - 7) : pairCount;
  const tokenPool = shuffle([...alphabet, ...alphabet,
    ...Array.from({length: pairCount - 10}, () => alphabet[randomInt(5)])]);
  for (let i = 0; i < pairCount; i++) {
    const j = i < firstCount ? 0 : 1;
    const s = parts[j], at = randomInt(s.length + 1), c = tokenPool[i];
    parts[j] = s.slice(0, at) + c + c + s.slice(at);
  }
  return parts.join(' ');
}

function engine() {
  const cache = new Map();
  function moves(state) {
    if (cache.has(state)) return cache.get(state);
    const result = new Map();
    for (const match of state.matchAll(/([A-E])\1+/g)) {
      const c = match[1], size = match[0].length, start = match.index;
      if (!result.has(c)) result.set(c, {size,
        whole: state.slice(0, start) + state.slice(start + size),
        pair: state.slice(0, start) + state.slice(start + 2)});
    }
    cache.set(state, result);
    return result;
  }
  return moves;
}

function kind(a, b) {
  if (!b) return null;
  if (a.deletion_rule !== b.deletion_rule) return 'rule';
  if (a.priority !== b.priority) return 'condition';
  return 'goal';
}
function diagnostic(type, action, choices, round, next) {
  if (type === 'rule') return choices.get(action).size > 2;
  return type === 'condition' && action === round.priority && choices.has(next.priority);
}

// All scores below are over successful action strings. The whole-mode maximum
// step count will be compared to the exactly known pair-mode step count.
function solve(start, round, next, moves, exact = null) {
  const memo = new Map(), type = kind(round, next);
  function visit(state) {
    if (memo.has(state)) return memo.get(state);
    const end = terminal(state, round.target_length);
    if (end === 'goal' && (exact === null || state === exact)) {
      return {count: 1, low: 0, high: 0, longest: 0, path: ''};
    }
    if (length(state) < round.target_length + 2) return null;
    const choices = moves(state);
    const allowed = choices.has(round.priority) ? [round.priority] : choices.keys();
    let result = null;
    for (const action of allowed) {
      const child = visit(choices.get(action)[round.deletion_rule]);
      if (!child) continue;
      if (!result) result = {count: 0, low: Infinity, high: -1, longest: 0, path: ''};
      const hit = Number(diagnostic(type, action, choices, round, next));
      result.count += child.count;
      result.longest = Math.max(result.longest, child.longest + 1);
      result.low = Math.min(result.low, child.low + hit);
      if (child.high + hit > result.high) {
        result.high = child.high + hit;
        result.path = action + child.path;
      }
    }
    memo.set(state, result);
    return result;
  }
  return visit(start);
}

function replay(start, round, answer, moves, next = null) {
  let state = start;
  const hits = [], states = [start], type = kind(round, next);
  for (const [i, action] of [...answer].entries()) {
    const end = terminal(state, round.target_length);
    const choices = moves(state);
    let reason = end === 'goal' ? 'continued_after_goal' : end;
    if (!reason && !choices.has(action)) reason = 'no_removable_run';
    if (!reason && choices.has(round.priority) && action !== round.priority) reason = 'priority_violation';
    if (reason) return {valid: false, step: reason === 'duplicate_final_pair' ? i : i + 1, reason, state, states, hits};
    if (diagnostic(type, action, choices, round, next)) hits.push(i + 1);
    state = choices.get(action)[round.deletion_rule];
    states.push(state);
  }
  const reason = terminal(state, round.target_length) ?? 'wrong_final_length';
  return {valid: reason === 'goal', step: answer.length, reason, state, states, hits};
}

function spaceProbe(input, rounds, moves) {
  for (const round of rounds) {
    const original = replay(input, round, round.one_valid_answer, moves);
    const flat = replay(lettersOnly(input), round, round.one_valid_answer, moves);
    if (flat.valid) continue;
    let first = flat.step;
    for (let i = 1; i < flat.states.length; i++) {
      if (lettersOnly(original.states[i]) !== flat.states[i]) { first = i; break; }
    }
    return {round: round.round, priority: round.priority, one_valid_answer: round.one_valid_answer,
      first_different_step: first, failure_step_without_space: flat.step,
      failure_without_space: flat.reason, state_without_space_at_failure: flat.state};
  }
  return null;
}

function evaluate(input, spec) {
  const segments = input.split(' ');
  if (spec.spaced && segments[0].at(-1) === segments[1][0]) return null;
  const moves = engine(), rounds = [], stats = [], traces = [];
  let exhibited = false;
  for (const [i, round] of spec.settings.entries()) {
    const next = spec.settings[i + 1], type = kind(round, next);
    const result = solve(input, round, next, moves);
    if (!result || ((type === 'rule' || type === 'condition') && result.low < 2)) return null;
    let path = result.path;
    if (spec.exactExample && !exhibited && round.target_length === 3) {
      const witness = solve(input, round, next, moves, 'AB A');
      if (witness) { exhibited = true; path = witness.path; }
    }
    const trace = replay(input, round, path, moves, next);
    if (!trace.valid) throw Error('Unreplayable solution');
    rounds.push({...round, one_valid_answer: path, final_state: trace.state, n_valid_answers: result.count});
    traces.push(trace); stats.push(result);
  }
  if (spec.exactExample && !exhibited) return null;
  const transitions = [];
  for (let i = 0; i < 4; i++) {
    const from = rounds[i], to = rounds[i + 1], type = kind(from, to);
    if (replay(input, to, from.one_valid_answer, moves).valid) return null;
    const t = {from_round: i + 1, to_round: i + 2, kind: type, reference_valid_under_new: false};
    if (type === 'rule') {
      const whole = from.deletion_rule === 'whole' ? stats[i] : stats[i + 1];
      const pairSteps = (length(input) - from.target_length) / 2;
      if (whole.longest >= pairSteps) return null;
      Object.assign(t, {old_rule: from.deletion_rule, new_rule: to.deletion_rule,
        effect_steps_on_old_reference: traces[i].hits, min_effect_steps_over_old_solutions: stats[i].low,
        max_whole_solution_steps: whole.longest, pair_solution_steps: pairSteps});
    } else if (type === 'condition') {
      Object.assign(t, {old_priority: from.priority, new_priority: to.priority,
        old_priority_matches_on_replayed_reference: traces[i].hits,
        min_over_all_previous_solutions: stats[i].low, max_over_all_previous_solutions: stats[i].high});
    } else {
      Object.assign(t, {old_target_length: from.target_length, new_target_length: to.target_length});
    }
    transitions.push(t);
  }
  const probe = spec.spaced ? spaceProbe(input, rounds, moves) : null;
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
    const round_settings = spec.settings.map(({round, target_length, deletion_rule}) => ({round, target_length, deletion_rule}));
    inputItems.push({id: spec.id, input, target_length: spec.target, round_settings});
    answerItems.push(answer); seen.add(input); found = true;
  }
  if (!found) throw Error('No qualifying input for ' + spec.id);
}
const version = 'pilot-3';
const problems = {
  version, rounds_per_item: 5,
  rules_en: 'An action names one letter A through E. Find its leftmost run of at least two identical letters within one segment. In whole mode, delete this entire run. In pair mode, delete just two letters from it; any remainder stays on the tape. After deletion the neighbors in that segment meet. If the priority letter has a removable run, you must select it. Otherwise any removable letter is allowed.',
  space_rule_en: 'A literal ASCII space is a permanent boundary. Letters on opposite sides cannot form a run. Ignore spaces only when counting or comparing the remaining letters for the goal.',
  space_example_en: 'In whole mode with priority B and target zero, AABBAA clears with BA, while "AABB AA" needs BAA because the two A runs stay separate.',
  goal_en: 'Stop as soon as the remaining letter count equals the round target. At target 2, the two remaining letters must differ: AB, BA or "A B" succeeds; AA or "A A" fails. At targets 0, 1 and 3 only the remaining count matters: even "AB A" counts as three. A remaining space alone counts as zero.',
  reset: 'Reset to the original input each round. Deliver only the current round settings and priority to participants. Use round_settings for the active target_length and deletion_rule. The item-level target_length is the base goal shared by four rounds. Submit the sequence of deletion letters in order.',
  items: inputItems,
};
const keys = {version, seed,
  evaluator_notes: 'Accept any physically valid action string that honors the round priority and reaches its goal on the final action. For R2 onward, previous_only means that the full submission is valid under the preceding round and invalid under this one. For priority-condition switches, also record old-letter choices when both priorities are available. For deletion-rule switches, effect_steps_on_old_reference identifies actions on the old trajectory that select runs of 3 or more: the same action has different effects under the two mechanics. These positions are opportunities, not evidence of an error by themselves. Whole-mode maximum solution length is strictly below the fixed pair-mode solution length at those switches. Space probes compare the same answer with and without a boundary under the same round settings. Steps are one-based.',
  items: answerItems,
};
for (const [name, content] of [['problems.json', problems], ['answer_key.json', keys]]) {
  writeFileSync(new URL(name, import.meta.url), JSON.stringify(content, null, 2) + '\n');
}
console.log('Generated ' + inputItems.length + ' pilot items.');
