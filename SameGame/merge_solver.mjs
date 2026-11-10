const alphabet = 'ABCDE';

function compileTape(input, target) {
  if (typeof input !== 'string' || /[^A-E ]/.test(input)) {
    throw new TypeError('The tape must contain only A–E and ASCII spaces.');
  }
  if (!Number.isInteger(target) || target < 0 || target > 3) {
    throw new RangeError('The target must be an integer from 0 through 3.');
  }
  const segments = input.split(' ');
  const size = input.length - segments.length + 1;
  if (size > 32) throw new RangeError('At most 32 letters are supported.');

  // A whole-run deletion never splits an original maximal run, so a bit
  // per original run retains provenance even when equal letters meet.
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
  const initial = (2 ** letters.length - 1) >>> 0;

  function render(mask) {
    const parts = Array(segments.length).fill('');
    for (let i = 0; i < letters.length; i++) {
      if (mask & (1 << i)) parts[segmentIds[i]] += alphabet[letters[i]].repeat(lengths[i]);
    }
    return parts.join(' ');
  }

  function moves(mask) {
    const options = Array(5);
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
      if (count < 2 || options[letter]) continue;
      const merged = (group & (group - 1)) !== 0;
      options[letter] = {
        letter: alphabet[letter], next: (mask & ~group) >>> 0, count,
        merged, singleton: merged && (group & nonSingletons) === 0,
      };
    }
    return options;
  }

  function isGoal(mask, remaining) {
    if (remaining !== target) return false;
    if (target !== 2) return true;
    const first = 31 - Math.clz32(mask & -mask);
    const rest = (mask & (mask - 1)) >>> 0;
    return rest !== 0 && letters[first] !== letters[31 - Math.clz32(rest & -rest)];
  }

  return {initial, size, moves, render, isGoal};
}

export function replayRound(input, target, answer) {
  const tape = compileTape(input, target);
  const merge_steps = [], singleton_merge_steps = [];
  let mask = tape.initial, remaining = tape.size, completed = 0;
  if (typeof answer === 'string') {
    for (const letter of answer) {
      if (remaining === target) break;
      const move = tape.moves(mask)[alphabet.indexOf(letter)];
      if (!move) break;
      completed++;
      if (move.merged) merge_steps.push(completed);
      if (move.singleton) singleton_merge_steps.push(completed);
      mask = move.next;
      remaining -= move.count;
    }
  }
  return {
    valid: typeof answer === 'string' && completed === answer.length && tape.isGoal(mask, remaining),
    final_state: tape.render(mask), merge_steps, singleton_merge_steps,
  };
}

export function solveRound(input, target, exactFinal = null) {
  const tape = compileTape(input, target);
  const memo = new Map();
  const dead = {count: 0}, terminal = {count: 1, minMerged: 0, minSingleton: 0};
  function search(mask, remaining) {
    if (memo.has(mask)) return memo.get(mask);
    let entry = dead;
    if (remaining === target) {
      if (tape.isGoal(mask, remaining)) {
        entry = {...terminal, witness: exactFinal === null || tape.render(mask) === exactFinal};
      }
    } else if (remaining >= target + 2) {
      let count = 0, minMerged = Infinity, minSingleton = Infinity, selected;
      for (const move of tape.moves(mask)) {
        if (!move) continue;
        const suffix = search(move.next, remaining - move.count);
        if (!suffix.count) continue;
        count += suffix.count;
        minMerged = Math.min(minMerged, Number(move.merged) + suffix.minMerged);
        minSingleton = Math.min(minSingleton, Number(move.singleton) + suffix.minSingleton);
        if (!selected && suffix.witness) selected = move;
      }
      if (count) entry = {count, minMerged, minSingleton, witness: Boolean(selected), selected};
    }
    memo.set(mask, entry);
    return entry;
  }

  const root = search(tape.initial, tape.size);
  if (!root.witness) return null;
  const merge_steps = [], singleton_merge_steps = [];
  let answer = '', mask = tape.initial;
  for (let move = root.selected; move; move = memo.get(mask).selected) {
    answer += move.letter;
    if (move.merged) merge_steps.push(answer.length);
    if (move.singleton) singleton_merge_steps.push(answer.length);
    mask = move.next;
  }
  return {
    one_valid_answer: answer, final_state: tape.render(mask), n_valid_answers: root.count,
    min_merged_deletions: root.minMerged, min_singleton_merge_deletions: root.minSingleton,
    merge_steps, singleton_merge_steps,
  };
}
