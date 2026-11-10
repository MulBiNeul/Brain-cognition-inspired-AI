import { writeFileSync } from 'node:fs';
import { pilotDesign } from './pilot_design.mjs';
import { replayRound, solveRound } from './merge_solver.mjs';

const version = 'pilot-5';
const seed = 2026111005;
const alphabet = 'ABCDE';
let randomState = seed;

function randomInt(limit) {
  randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
  return Math.floor(randomState / 2 ** 32 * limit);
}
function pick(values) {
  return values[randomInt(values.length)];
}
function shuffle(values) {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i--) {
    const j = randomInt(i + 1);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function makeTape(size, target, spaced, exactFinal) {
  const parts = exactFinal === null ? Array(spaced ? 2 : 1).fill('') : exactFinal.split(' ');
  if (exactFinal === null) {
    const finalLetters = target === 2 ? shuffle(alphabet).slice(0, 2)
      : Array.from({length: target}, () => pick(alphabet));
    for (const letter of finalLetters) parts[randomInt(parts.length)] += letter;
  }
  // Insert in reverse. Splitting two equal neighbors by a different run
  // creates a middle that can later be removed to reconnect the pair.
  const widths = [];
  let remaining = size - target;
  while (remaining > 0) {
    const width = remaining === 3 || (remaining >= 5 && randomInt(5) === 0) ? 3 : 2;
    widths.push(width);
    remaining -= width;
  }
  const firstSegmentGroups = spaced ? 3 + randomInt(widths.length - 5) : widths.length;
  for (const [i, width] of widths.entries()) {
    const part = i < firstSegmentGroups ? 0 : 1;
    const text = parts[part];
    const slots = [];
    for (let at = 1; at < text.length; at++) {
      if (text[at - 1] === text[at]) slots.push(at);
    }
    const at = slots.length && randomInt(3) !== 0 ? pick(slots) : randomInt(text.length + 1);
    const letter = pick([...alphabet].filter(c => c !== text[at - 1] && c !== text[at]));
    parts[part] = text.slice(0, at) + letter.repeat(width) + text.slice(at);
  }
  return parts.join(' ');
}

// Avoid reusing the same board with a different set of letter names.
function patternOf(input) {
  const names = new Map();
  return Array.from(input, c => {
    if (c === ' ') return c;
    if (!names.has(c)) names.set(c, alphabet[names.size]);
    return names.get(c);
  }).join('');
}

function generateRound(spec, index, size, seen) {
  const target = spec.targets[index];
  const exact = spec.id === 'SG-036' && index === 1 ? 'AB A' : null;
  for (let attempt = 0; attempt < 10000; attempt++) {
    const input = makeTape(size, target, spec.spaced, exact);
    if (![...alphabet].every(c => input.includes(c))) continue;
    const openers = new Set(Array.from(input.matchAll(/([A-E])\1+/g), match => match[1]));
    if (openers.size < 2) continue;
    const pattern = patternOf(input);
    if (seen.has(pattern)) continue;
    const solution = solveRound(input, target, exact);
    if (!solution || solution.min_merged_deletions < 2 || solution.min_singleton_merge_deletions < 1) continue;
    if (spec.spaced && replayRound(input.replace(' ', ''), target, solution.one_valid_answer).valid) continue;
    seen.add(pattern);
    return {input, solution};
  }
  throw new Error('Could not generate ' + spec.id + ' round ' + (index + 1));
}

const publicItems = [], answerItems = [], seen = new Set();
const sizes = shuffle(Array.from({length: pilotDesign.length * 5}, (_, i) => 28 + i % 5));
const schedule = ['# SameGame 파일럿의 라운드별 입력과 목표', '',
  '각 행의 정답은 삭제할 글자 순서의 한 예시다. 뒤의 두 칸은 가능한 모든 정답에서 각각 구한 최소 횟수다. 병합 삭제는 원래 떨어져 있던 같은 글자가 붙은 뒤 그 묶음을 지우는 행동이다. 단독 글자 병합 삭제는 이 중 원래 한 글자씩만 있던 부분들로 이루어진 묶음을 지우는 행동이다. 공백은 입력에 표시된 그대로 영구 경계다.', '',
  '| 문제 | 라운드 | 입력 | 목표 | 정답 예시 | 최소 병합 삭제 | 최소 단독 글자 병합 삭제 |',
  '|---|---:|---|---:|---|---:|---:|'];
for (const spec of pilotDesign) {
  const round_settings = [], rounds = [];
  let space_probe = null;
  for (const [i, target_length] of spec.targets.entries()) {
    const round = i + 1;
    const {input, solution} = generateRound(spec, i, sizes.pop(), seen);
    round_settings.push({round, input, target_length});
    rounds.push({round, ...solution});
    if (spec.spaced && !space_probe) space_probe = {round, answer: solution.one_valid_answer};
    schedule.push('| ' + [spec.id, round, '`' + input + '`', target_length, '`' + solution.one_valid_answer + '`',
      solution.min_merged_deletions, solution.min_singleton_merge_deletions].join(' | ') + ' |');
  }
  publicItems.push({id: spec.id, round_settings});
  answerItems.push({id: spec.id, rounds, space_probe});
}

const problems = {
  version, rounds_per_item: 5,
  rules_en: 'Each action names one letter from A to E. Choose any letter with a run of at least two identical adjacent letters. Delete its entire leftmost eligible run. Neighbors in that segment then meet. A newly formed run remains on the tape until you select its letter in a later action.',
  space_rule_en: 'A literal ASCII space is a permanent boundary. Letters on opposite sides cannot form a run. Ignore spaces when counting and comparing the remaining letters for the goal.',
  goal_en: 'Stop when the remaining letter count first equals the round target. For target 2 the two letters must differ: AB and "A B" succeed, while AA and "A A" fail. For targets 0, 1 and 3 only the count matters. "AB A" counts as three and a space alone counts as zero.',
  reset: 'Each round starts from the new input in its own round_settings entry. Present only that entry for the current round. Submit the letters selected for deletion, in order.',
  items: publicItems,
};
const keys = {version, seed,
  evaluator_notes: 'Accept every legal action string that first reaches the required goal on its final action. Merge counts describe the construction of the board and need no extra scoring rule. A merged deletion removes blocks from two or more original runs. A singleton merge deletion is one of these where every original run was one letter long. Both minima are computed independently over all accepted sequences. Step indices are one-based. Each spaced round has a reference answer that fails when replayed with the boundary removed.',
  items: answerItems,
};
for (const [name, data] of [['problems.json', problems], ['answer_key.json', keys]]) {
  writeFileSync(new URL(name, import.meta.url), JSON.stringify(data, null, 2) + '\n');
}
writeFileSync(new URL('ROUND_SCHEDULE.md', import.meta.url), schedule.join('\n') + '\n');
console.log('Generated ' + publicItems.length + ' items with ' + seen.size + ' distinct round inputs.');
