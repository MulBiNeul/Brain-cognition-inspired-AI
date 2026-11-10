import { writeFileSync } from 'node:fs';
import { studyDesign } from './pilot_design.mjs';

const alphabet = 'ABCDE';
const seed = 2026111004;
let randomState = seed;
function randomInt(limit) {
  randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
  return Math.floor(randomState / 2 ** 32 * limit);
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
function terminal(state, target) {
  const letters = lettersOnly(state);
  if (letters.length !== target) return null;
  return target === 2 && letters[0] === letters[1] ? 'duplicate_final_pair' : 'goal';
}

// Work backwards by inserting removable groups. Forward replay then checks
// all five rounds, including priorities and the direction of each selection.
function makeInput(spec) {
  const parts = spec.spaced ? ['', ''] : [''];
  if (spec.exactExample) {
    parts[0] = 'AB';
    parts[1] = 'A';
  } else {
    for (const c of shuffle(alphabet).slice(0, randomInt(4))) parts[randomInt(parts.length)] += c;
  }
  const size = 28 + randomInt(5);
  let remaining = size - parts.join('').length;
  const groups = [];
  while (remaining > 0) {
    const width = remaining === 3 || (remaining > 4 && randomInt(5) === 0) ? 3 : 2;
    groups.push(width);
    remaining -= width;
  }
  const nFirst = spec.spaced ? 3 + randomInt(groups.length - 5) : groups.length;
  const pool = shuffle([...alphabet, ...alphabet,
    ...Array.from({length: Math.max(0, groups.length - 10)}, () => alphabet[randomInt(5)])]);
  for (const [i, width] of groups.entries()) {
    const j = i < nFirst ? 0 : 1;
    const s = parts[j], at = randomInt(s.length + 1);
    parts[j] = s.slice(0, at) + pool[i].repeat(width) + s.slice(at);
  }
  return parts.join(' ');
}

function engine() {
  const cache = new Map();
  return state => {
    if (cache.has(state)) return cache.get(state);
    const left = new Map(), right = new Map();
    for (const match of state.matchAll(/([A-E])\1+/g)) {
      const letter = match[1], start = match.index;
      const after = state.slice(0, start) + state.slice(start + match[0].length);
      if (!left.has(letter)) left.set(letter, after);
      right.set(letter, after);
    }
    const options = {left, right};
    cache.set(state, options);
    return options;
  };
}

function transitionKind(from, to) {
  if (from.selection_side !== to.selection_side) return 'goal_and_rule';
  if (from.priority !== to.priority) return 'goal_and_condition';
  return 'goal';
}
function opportunity(round, next, action, options) {
  if (!next) return false;
  if (round.selection_side !== next.selection_side) return options.left.get(action) !== options.right.get(action);
  return round.priority !== next.priority && action === round.priority && options.left.has(next.priority);
}
function permitted(choices, priority) {
  return choices.has(priority) ? [priority] : choices.keys();
}

// Count all accepted action strings on the DAG of progressively shorter tapes.
function solve(input, round, next, moves, exact = null) {
  const memo = new Map();
  function visit(state) {
    if (memo.has(state)) return memo.get(state);
    const end = terminal(state, round.target_length);
    if (end === 'goal' && (exact === null || state === exact)) return {count: 1, low: 0, high: 0, path: ''};
    if (lettersOnly(state).length < round.target_length + 2) return null;
    const options = moves(state), choices = options[round.selection_side];
    let result = null;
    for (const action of permitted(choices, round.priority)) {
      const child = visit(choices.get(action));
      if (!child) continue;
      const hit = Number(opportunity(round, next, action, options));
      if (!result) result = {count: 0, low: Infinity, high: -1, path: ''};
      result.count += child.count;
      result.low = Math.min(result.low, child.low + hit);
      if (child.high + hit > result.high) {
        result.high = child.high + hit;
        result.path = action + child.path;
      }
    }
    memo.set(state, result);
    return result;
  }
  return visit(input);
}

function replay(input, round, answer, moves, next = null) {
  let state = input;
  const hits = [], states = [input];
  for (const [i, action] of [...answer].entries()) {
    const end = terminal(state, round.target_length), options = moves(state), choices = options[round.selection_side];
    let reason = end === 'goal' ? 'continued_after_goal' : end;
    if (!reason && !choices.has(action)) reason = 'no_removable_run';
    if (!reason && choices.has(round.priority) && action !== round.priority) reason = 'priority_violation';
    if (reason) return {valid: false, step: reason === 'duplicate_final_pair' ? i : i + 1, reason, state, states, hits};
    if (opportunity(round, next, action, options)) hits.push(i + 1);
    state = choices.get(action);
    states.push(state);
  }
  const reason = terminal(state, round.target_length) ?? 'wrong_final_length';
  return {valid: reason === 'goal', step: answer.length, reason, state, states, hits};
}

// The grader records priority violations but can still follow a physically
// executable submission until the current round ends or a named run is absent.
function currentOpportunities(input, from, to, answer, moves) {
  let state = input, hits = 0;
  for (const action of answer) {
    if (terminal(state, to.target_length)) break;
    const options = moves(state), choices = options[to.selection_side];
    if (!choices.has(action)) break;
    if (opportunity(from, to, action, options)) hits++;
    state = choices.get(action);
  }
  return hits;
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

// Find a submission that follows the NEW goal and only the OLD non-goal
// setting. Rule witnesses can diverge physically, so replay both competing
// goal interpretations on their own tapes and reject ambiguous answers.
function carryoverProbe(input, from, to, moves) {
  const desired = {...from, target_length: to.target_length};
  const otherGoal = {...to, target_length: from.target_length};
  const exhausted = new Set();
  function advance(state, round, action) {
    if (state === null || terminal(state, round.target_length)) return null;
    const choices = moves(state)[round.selection_side];
    if (choices.has(round.priority) && round.priority !== action) return null;
    return choices.get(action) ?? null;
  }
  function search(state, current, revertedGoal) {
    const key = [state, current ?? '-', revertedGoal ?? '-'].join('|');
    if (exhausted.has(key)) return null;
    if (terminal(state, desired.target_length) === 'goal') {
      const accept = (s, round) => s !== null && terminal(s, round.target_length) === 'goal';
      return accept(current, to) || accept(revertedGoal, otherGoal) ? null : '';
    }
    if (lettersOnly(state).length < desired.target_length + 2) return null;
    const choices = moves(state)[desired.selection_side];
    for (const action of permitted(choices, desired.priority)) {
      const tail = search(choices.get(action), advance(current, to, action), advance(revertedGoal, otherGoal, action));
      if (tail !== null) return action + tail;
    }
    exhausted.add(key);
    return null;
  }
  const answer = search(input, input, input);
  return answer === null ? null : {answer};
}

function evaluate(input, spec) {
  const moves = engine(), rounds = [], stats = [], traces = [];
  let exhibited = false;
  for (const [i, round] of spec.settings.entries()) {
    const next = spec.settings[i + 1];
    const result = solve(input, round, next, moves);
    if (!result || (next && transitionKind(round, next) !== 'goal' && result.low < 2)) return null;
    let path = result.path;
    if (spec.exactExample && !exhibited && round.target_length === 3) {
      const witness = solve(input, round, next, moves, 'AB A');
      if (witness) { exhibited = true; path = witness.path; }
    }
    const trace = replay(input, round, path, moves, next);
    if (!trace.valid) throw Error('Unreplayable reference solution');
    rounds.push({...round, one_valid_answer: path, final_state: trace.state, n_valid_answers: result.count});
    traces.push(trace);
    stats.push(result);
  }
  if (spec.exactExample && !exhibited) return null;
  const transitions = [];
  for (let i = 0; i < 4; i++) {
    const from = rounds[i], to = rounds[i + 1], kind = transitionKind(from, to);
    if (replay(input, to, from.one_valid_answer, moves).valid) return null;
    const t = {from_round: from.round, to_round: to.round, kind, old_target_length: from.target_length,
      new_target_length: to.target_length, reference_valid_under_new: false};
    if (kind === 'goal_and_rule') {
      Object.assign(t, {old_selection_side: from.selection_side, new_selection_side: to.selection_side,
        effect_steps_on_old_reference: traces[i].hits, min_effect_steps_over_old_solutions: stats[i].low,
        max_effect_steps_over_old_solutions: stats[i].high});
    } else if (kind === 'goal_and_condition') {
      Object.assign(t, {old_priority: from.priority, new_priority: to.priority,
        old_priority_matches_on_replayed_reference: traces[i].hits,
        min_over_all_previous_solutions: stats[i].low, max_over_all_previous_solutions: stats[i].high});
    }
    if (kind !== 'goal') {
      t.carryover_probe = carryoverProbe(input, from, to, moves);
      if (!t.carryover_probe || [from.one_valid_answer, t.carryover_probe.answer].some(
        answer => currentOpportunities(input, from, to, answer, moves) < 2)) return null;
    }
    transitions.push(t);
  }
  const probe = spec.spaced ? spaceProbe(input, rounds, moves) : null;
  if (spec.spaced && !probe) return null;
  return {id: spec.id, rounds, transitions, space_probe: probe};
}

const design = studyDesign(spec => Boolean(evaluate(spec.input, spec)));
const inputItems = [], answerItems = [], seen = new Set();
for (const spec of design) {
  let found = false;
  for (let attempt = 0; attempt < 250000 && !found; attempt++) {
    const input = attempt === 0 ? spec.input : makeInput(spec);
    if (seen.has(input)) continue;
    const answer = evaluate(input, spec);
    if (!answer) continue;
    inputItems.push({id: spec.id, input, target_length: spec.settings[0].target_length, round_settings: spec.settings});
    answerItems.push(answer);
    seen.add(input);
    found = true;
  }
  if (!found) throw Error('No qualifying input for ' + spec.id);
}
const version = 'pilot-4';
const problems = {version, rounds_per_item: 5,
  rules_en: 'An action names one letter A through E. A run consists of two or more adjacent copies of that letter within the same segment. Delete the entire selected run, including all three letters of AAA. When the named letter has multiple eligible runs, select the leftmost or rightmost one as specified by selection_side for this round. The remaining neighbors in that segment meet. If the priority letter has an eligible run, you must select it. Otherwise choose any letter with an eligible run.',
  space_rule_en: 'A literal ASCII space is a permanent boundary. Letters on opposite sides cannot form a run. Ignore spaces when counting the remaining letters and checking whether the last two letters differ.',
  space_example_en: 'With priority B and target zero, AABBAA clears with BA, while "AABB AA" needs BAA because the two A runs stay separate.',
  goal_en: 'Stop as soon as the remaining letter count equals the current target. At target 2 the two letters must differ: AB, BA and "A B" succeed, while AA and "A A" fail. At targets 0, 1 and 3 only the count matters. "AB A" and AAA each count as three. A remaining space alone counts as zero.',
  reset: 'Reset to the original input at the start of each round. Give the participant the current round_settings entry only. Its target_length is the required residual letter count. The item-level target_length is the first-round value. Submit the sequence of deletion letters in order.',
  items: inputItems,
};
const keys = {version, seed,
  evaluator_notes: 'Replay any submission under the current settings and stop when it first reaches the target. For R2 onward, previous_only means valid under the full preceding round and invalid under the current one. At joint switches separately revert only the goal or only the changed non-goal setting. Each carryover_probe succeeds only with the new goal and the previous non-goal setting. One-based reference opportunity steps are counted along the old solution until its own endpoint. An opportunity on a reference is not an observed participant error.',
  items: answerItems,
};
for (const [name, content] of [['problems.json', problems], ['answer_key.json', keys]]) {
  writeFileSync(new URL(name, import.meta.url), JSON.stringify(content, null, 2) + '\n');
}

const schedule = ['# SameGame 50문제의 이전 설정과 현재 설정', '',
  '각 칸은 왼쪽부터 R1, R2, R3, R4, R5 순서다. 두 버전의 문제 ID, 공백 유무, 우선 글자는 일치한다. 공개 입력과 정답 예시는 각각 [problems.json](problems.json)과 [answer_key.json](answer_key.json)에 있다.', '',
  'v3 삭제에서 W는 묶음 전체 삭제, P는 그 묶음에서 두 개만 삭제다. v3는 항상 왼쪽 묶음을 선택했다. v4는 어느 방향에서 골라도 묶음 전체를 삭제한다. L은 선택한 글자의 왼쪽 끝 묶음, R은 오른쪽 끝 묶음이다.', '',
  '| 문제 | 공백 | v3 목표 | v3 삭제 | 우선 글자 (공통) | v4 목표 | v4 방향 | 입력 문자열 |',
  '|---|---|---|---|---|---|---|---|'];
for (const [i, item] of design.entries()) {
  const cells = [item.id, item.spaced ? '있음' : '없음', item.v3Goals.join(' · '),
    item.v3Rules.map(r => r === 'whole' ? 'W' : 'P').join(' · '), item.settings.map(r => r.priority).join(' · '),
    item.settings.map(r => r.target_length).join(' · '), item.settings.map(r => r.selection_side === 'left' ? 'L' : 'R').join(' · '),
    item.input === inputItems[i].input ? '유지' : '재생성'];
  schedule.push('| ' + cells.join(' | ') + ' |');
}
writeFileSync(new URL('ROUND_SCHEDULE.md', import.meta.url), schedule.join('\n') + '\n');
console.log('Generated ' + inputItems.length + ' items with five verified round solutions each.');
