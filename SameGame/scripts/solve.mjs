const alphabet = 'ABCDE';
const noLetter = alphabet.length;

function ruleCode(rule) {
  if (rule == null) return -1;
  if (typeof rule !== 'string' || !/^[A-E]{2}$/.test(rule)) {
    throw new TypeError('A forbidden pair must consist of two A–E letters.');
  }
  return 5 * alphabet.indexOf(rule[0]) + alphabet.indexOf(rule[1]);
}

function compile(input, target) {
  if (typeof input !== 'string' || /[^A-E ]/.test(input)) throw new TypeError('Invalid tape.');
  if (!Number.isInteger(target) || target < 0 || target > 3) throw new RangeError('Invalid target.');
  const segments = input.split(' ');
  const size = input.length - segments.length + 1;
  if (size > 32) throw new RangeError('At most 32 letters are supported.');
  const letters = [], lengths = [], segmentIds = [];
  let nonSingletons = 0;
  segments.forEach((segment, id) => {
    for (const match of segment.matchAll(/([A-E])\1*/g)) {
      if (match[0].length > 1) nonSingletons |= 1 << letters.length;
      letters.push(alphabet.indexOf(match[1]));
      lengths.push(match[0].length);
      segmentIds.push(id);
    }
  });

  // Whole-run deletion preserves each original maximal run as a unit.
  // Distinct bit masks can therefore retain different provenance for the
  // same displayed string. Segment IDs preserve even an empty boundary.
  const initial = (2 ** letters.length - 1) >>> 0;
  const states = new Map();
  function moves(mask) {
    if (states.has(mask)) return states.get(mask);
    const result = [], ordinals = Array(5).fill(0);
    let pending = mask;
    while (pending) {
      const first = 31 - Math.clz32(pending & -pending);
      const letter = letters[first], segment = segmentIds[first];
      let group = 0, count = 0, i = first;
      do {
        const bit = pending & -pending;
        group |= bit;
        count += lengths[i];
        pending = (pending ^ bit) >>> 0;
        i = pending ? 31 - Math.clz32(pending & -pending) : -1;
      } while (i >= 0 && letters[i] === letter && segmentIds[i] === segment);
      if (count < 2) continue;
      const ordinal = ++ordinals[letter];
      const merged = (group & (group - 1)) !== 0;
      result.push({letter, ordinal, token: alphabet[letter] + ordinal, next: (mask & ~group) >>> 0, count,
        merged, singleton: merged && (group & nonSingletons) === 0});
    }
    result.sort((a, b) => a.letter - b.letter);
    states.set(mask, result);
    return result;
  }
  function render(mask) {
    const parts = Array(segments.length).fill('');
    for (let i = 0; i < letters.length; i++) {
      if (mask & (1 << i)) parts[segmentIds[i]] += alphabet[letters[i]].repeat(lengths[i]);
    }
    return parts.join(' ');
  }
  function isGoal(mask, remaining) {
    return remaining === target && moves(mask).length === 0;
  }
  return {size, target, initial, moves, render, isGoal};
}

// Ordinals are re-counted per letter on each live board. Plain letter-only
// plans retain the original convention of selecting ordinal one throughout.
export function parseMoves(answer) {
  if (typeof answer !== 'string') return null;
  if (/^[A-E]+$/.test(answer)) return [...answer].map(letter => letter + '1');
  if (!/^[A-E][1-9][0-9]*(?: [A-E][1-9][0-9]*)*$/.test(answer)) return null;
  return answer.split(' ');
}

function trace(board, answer, rule, onViolation) {
  let mask = board.initial, remaining = board.size, last = noLetter, steps = 0;
  const merged_steps = [], singleton_merge_steps = [], violation_steps = [];
  const tokens = parseMoves(answer);
  for (const token of tokens ?? []) {
    const id = alphabet.indexOf(token[0]);
    if (5 * last + id === rule) {
      violation_steps.push(steps + 1);
      onViolation?.(mask, remaining, last, steps + 1);
    }
    const move = board.moves(mask).find(move => move.token === token);
    if (!move) break;
    steps++;
    if (move.merged) merged_steps.push(steps);
    if (move.singleton) singleton_merge_steps.push(steps);
    mask = move.next;
    remaining -= move.count;
    last = id;
  }
  return {
    valid: tokens !== null && steps === tokens.length && !violation_steps.length && board.isGoal(mask, remaining),
    final_state: board.render(mask).trim(), merged_steps, singleton_merge_steps, violation_steps,
  };
}

export function replay(input, target, answer, forbidden) {
  return trace(compile(input, target), answer, ruleCode(forbidden));
}

// Choose a legal letter uniformly, then a removable run of that letter uniformly. For counting
// successes only, discard branches that cannot end at the required length:
// deletions always reduce length, and a valid final has no removable run.
export function uniformPlayOutcomes(input, target, forbidden) {
  const board = compile(input, target), rule = ruleCode(forbidden);
  const first = Math.floor(rule / 5), second = rule % 5;
  const layers = Array.from({length: board.size + 1}, () => new Map());
  const outcomes = new Map();
  let successfulMoves = 0;
  layers[board.size].set(2 * board.initial, {probability: 1, paths: 1});
  for (let remaining = board.size; remaining >= target; remaining--) {
    for (const [key, flow] of layers[remaining]) {
      const mask = Math.floor(key / 2), followsFirst = key % 2 === 1;
      if (remaining === target) {
        if (board.isGoal(mask, remaining)) {
          const letters = [...board.render(mask).replaceAll(' ', '')].sort().join('');
          outcomes.set(letters, (outcomes.get(letters) ?? 0) + flow.probability);
          successfulMoves += flow.paths;
        }
        continue;
      }
      const moves = board.moves(mask).filter(move => !(followsFirst && move.letter === second));
      const runCounts = Array(5).fill(0);
      for (const move of moves) runCounts[move.letter]++;
      const letterCount = runCounts.filter(count => count > 0).length;
      for (const move of moves) {
        const nextRemaining = remaining - move.count;
        if (nextRemaining < target) continue;
        const next = 2 * move.next + Number(move.letter === first);
        const layer = layers[nextRemaining];
        const mass = flow.probability / letterCount / runCounts[move.letter];
        const combined = layer.get(next);
        if (combined) {
          combined.probability += mass;
          combined.paths += flow.paths;
        } else {
          layer.set(next, {probability: mass, paths: flow.paths});
        }
      }
    }
  }
  const masses = [...outcomes.values()];
  const success = masses.reduce((sum, value) => sum + value, 0);
  return {success, mostCommonGivenSuccess: success ? Math.max(...masses) / success : 0,
    successfulMoves, multisetCount: outcomes.size};
}

// Every move deletes at least two letters. A search for an exact-length
// success may prune shorter branches without changing physical replay.
function unrestrictedMinima(board) {
  const memo = new Map();
  return function minimum(mask, remaining) {
    if (memo.has(mask)) return memo.get(mask);
    let result = null;
    if (board.isGoal(mask, remaining)) {
      result = [0, 0];
    } else if (remaining >= board.target + 2) {
      let merged = Infinity, singleton = Infinity;
      for (const move of board.moves(mask)) {
        const tail = minimum(move.next, remaining - move.count);
        if (!tail) continue;
        merged = Math.min(merged, Number(move.merged) + tail[0]);
        singleton = Math.min(singleton, Number(move.singleton) + tail[1]);
        if (merged === 0 && singleton === 0) break;
      }
      if (merged < Infinity) result = [merged, singleton];
    }
    memo.set(mask, result);
    return result;
  };
}

// Follow physically possible selections under the current pair until no
// removable run remains. A terminal of a different small length is both a
// plausible alternative hidden goal and a failed branch for this round.
function hasOutcomeChoices(board, rule, minimumMultisets) {
  const seen = new Set(), answers = new Set();
  const first = Math.floor(rule / 5);
  const required = Math.max(board.target === 0 ? 1 : 2, minimumMultisets);
  let alternativeLength = false;
  function visit(mask, remaining, last) {
    const key = 2 * mask + Number(last === first);
    if (seen.has(key)) return false;
    seen.add(key);
    const moves = board.moves(mask);
    if (!moves.length) {
      if (remaining === board.target) answers.add([...board.render(mask).replaceAll(' ', '')].sort().join(''));
      else if (remaining <= 3) alternativeLength = true;
      return answers.size >= required && alternativeLength;
    }
    for (const move of moves) {
      if (5 * last + move.letter !== rule &&
          visit(move.next, remaining - move.count, move.letter)) return true;
    }
    return false;
  }
  return visit(board.initial, board.size, noLetter);
}

function prepend(token, suffix) {
  return suffix ? token + ' ' + suffix : token;
}

function legalFinish(board, minimum, rule, exactFinal, requiredPair = -1, firstOnly = false, flat = null) {
  const memo = new Map();
  return function finish(mask, remaining, last, pending = requiredPair >= 0,
    flatMask = flat?.initial, flatRemaining = flat?.size) {
    const state = (mask * 6 + last) * 2 + Number(pending);
    const key = flat ? state + '/' + flatMask : state;
    if (memo.has(key)) return memo.get(key);
    let path = null;
    if (board.isGoal(mask, remaining)) {
      const finalMatches = exactFinal === null || board.render(mask).trim() === exactFinal;
      const boundaryMatters = !flat || flatMask === null || !flat.isGoal(flatMask, flatRemaining);
      if (!pending && finalMatches && boundaryMatters) path = '';
    } else if (remaining >= board.target + 2 && minimum(mask, remaining)) {
      for (const move of board.moves(mask)) {
        const pair = 5 * last + move.letter;
        if (pair === rule || (firstOnly && move.ordinal !== 1)) continue;
        let nextFlat = null, nextFlatRemaining = 0;
        if (flat && flatMask !== null) {
          const flatMove = flat.moves(flatMask).find(other => other.token === move.token);
          if (flatMove) {
            nextFlat = flatMove.next;
            nextFlatRemaining = flatRemaining - flatMove.count;
          }
        }
        const tail = finish(move.next, remaining - move.count, move.letter,
          pending && pair !== requiredPair, nextFlat, nextFlatRemaining);
        if (tail !== null) { path = prepend(move.token, tail); break; }
      }
    }
    memo.set(key, path);
    return path;
  };
}

function previousProbe(board, minimum, current, previous, canFinish) {
  if (current === previous || current < 0) return null;
  const memo = new Map();
  const finished = {answer: '', steps: 0, recoverable: 0, hits: 0};
  function best(mask, remaining, last, needed) {
    // 32-bit masks and their small integer suffix fit exactly in a JS Number.
    const key = (mask * 6 + last) * 3 + needed;
    if (memo.has(key)) return memo.get(key);
    let winner = null;
    if (board.isGoal(mask, remaining)) {
      if (needed === 0) winner = finished;
    } else if (remaining >= board.target + 2 * Math.max(needed, 1) && minimum(mask, remaining)) {
      for (const move of board.moves(mask)) {
        const pair = 5 * last + move.letter;
        if (pair === previous) continue;
        const hit = Number(pair === current);
        const tail = best(move.next, remaining - move.count, move.letter, Math.max(0, needed - hit));
        if (!tail) continue;
        const hits = hit + tail.hits;
        // On a forbidden step, a legal completion from this exact state and
        // previous action must begin with a different, currently legal move.
        const fresh = hit && canFinish(mask, remaining, last) !== null;
        const recoverable = Number(fresh) + tail.recoverable;
        if (!winner || recoverable > winner.recoverable ||
            (recoverable === winner.recoverable && (hits > winner.hits ||
             (hits === winner.hits && tail.steps + 1 < winner.steps)))) {
          winner = {answer: prepend(move.token, tail.answer), steps: tail.steps + 1, recoverable, hits};
        }
      }
    }
    memo.set(key, winner);
    return winner;
  }
  const probe = best(board.initial, board.size, noLetter, 2);
  if (!probe) return null;
  const recoverable_violation_steps = [];
  const result = trace(board, probe.answer, current, (mask, remaining, last, step) => {
    if (canFinish(mask, remaining, last) !== null) recoverable_violation_steps.push(step);
  });
  return {answer: probe.answer, steps: probe.steps, violation_steps: result.violation_steps, recoverable_violation_steps};
}

export function analyze(input, target, forbidden, previous = null, exactFinal = null, contrast = false, minimumMultisets = 0, requireBoundary = false, preferFirst = false) {
  const board = compile(input, target);
  const rule = ruleCode(forbidden), old = ruleCode(previous);
  const minimum = unrestrictedMinima(board);
  const stats = minimum(board.initial, board.size);
  if (!stats || stats[0] < 4 || stats[1] < 2) return null;
  if (!hasOutcomeChoices(board, rule, minimumMultisets)) return null;
  const anyFinish = legalFinish(board, minimum, rule, null);
  const flat = requireBoundary && input.includes(' ') ? compile(input.replace(' ', ''), target) : null;
  const required = contrast ? old : -1;
  let answer = null;
  // Prefer the original ordinal-one witness when it still meets every criterion.
  if (preferFirst) {
    answer = legalFinish(board, minimum, rule, exactFinal, required, true, flat)(board.initial, board.size, noLetter);
  }
  if (answer === null) {
    const reference = !flat && !contrast && exactFinal === null ? anyFinish
      : legalFinish(board, minimum, rule, exactFinal, required, false, flat);
    answer = reference(board.initial, board.size, noLetter);
  }
  if (answer === null) return null;
  const probe = old < 0 ? null : previousProbe(board, minimum, rule, old, anyFinish);
  if (old >= 0 && !probe) return null;
  const {final_state, merged_steps, singleton_merge_steps} = trace(board, answer, rule);
  return {answer, final_state, merged_steps, singleton_merge_steps, previous_rule_probe: probe,
    min_merged_deletions: stats[0], min_singleton_merge_deletions: stats[1]};
}
