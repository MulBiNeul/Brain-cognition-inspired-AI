function groupsOnTape(text) {
  const counts = Object.create(null);
  return Array.from(text.matchAll(/([A-E])\1*| /g), match => {
    const chars = match[0], letter = chars[0];
    let token = null;
    if (chars.length >= 2) {
      counts[letter] = (counts[letter] || 0) + 1;
      token = letter + counts[letter];
    }
    return {chars, start: match.index, token};
  });
}
function parsePlan(text) {
  if (typeof text !== 'string' || !text || /[^A-E0-9 ]/.test(text)) return null;
  if (/^[A-E]+$/.test(text)) return Array.from(text, letter => letter + '1');
  const tokens = text.split(' ');
  return tokens.every(token => /^[A-E][1-9][0-9]*$/.test(token)) ? tokens : null;
}
function inspectorReplay(input, target, forbidden, previous, moves, answer) {
  const states = [{
    text: input, action: null, removedStart: null, removedLength: 0, violation: false, executed: false
  }];
  const hasPrevious = typeof previous === "string" && previous.length === 2;
  const plan = parsePlan(moves);
  const validFormat = plan !== null && plan.length >= 1 && plan.length <= 30 && typeof answer === "string";
  const result = {
    validFormat, feedback: "Incorrect, but no rule violation", ruleOutcome: hasPrevious ? "ineligible" : null,
    persistentError: false, eligible: false,
    finalState: null, completed: false, remainingChoices: [], goalMet: null, physicalGoalMet: null, finalMatches: null,
    violationSteps: [], failedStep: null, states, reason: "invalid_format"
  };
  if (!validFormat) return result;

  let text = input;
  let previousViolation = false;
  // Count a recognizable forbidden pair even on the first impossible command.
  // A hidden target never stops the physical replay of a submitted plan.
  for (let i = 0; i < plan.length; i++) {
    const action = plan[i];
    const pair = i === 0 ? null : plan[i - 1][0] + action[0];
    const violation = pair !== null && pair === forbidden;
    if (violation) result.violationSteps.push(i + 1);
    if (hasPrevious && pair === previous) previousViolation = true;

    const group = groupsOnTape(text).find(run => run.token === action);
    if (!group) {
      result.failedStep = i + 1;
      states.push({text, action, removedStart: null, removedLength: 0, violation, executed: false});
      break;
    }
    const removedStart = group.start;
    const removedLength = group.chars.length;
    text = text.slice(0, removedStart) + text.slice(removedStart + removedLength);
    states.push({text, action, removedStart, removedLength, violation, executed: true});
  }

  // Only the outer ASCII boundary is omitted from a reported final answer.
  const canonical = value => value.replace(/^ +| +$/g, "");
  result.finalState = canonical(text);
  result.remainingChoices = groupsOnTape(text).filter(run => run.token).map(run => run.token);
  result.completed = result.failedStep === null && result.remainingChoices.length === 0;
  if (result.failedStep !== null) {
    result.reason = "no_removable_run";
  } else {
    const count = text.replace(/ /g, "").length;
    result.physicalGoalMet = result.completed && count === target;
    result.finalMatches = result.finalState === canonical(answer);
    result.goalMet = result.physicalGoalMet && result.finalMatches;
    if (!result.completed) result.reason = "incomplete_plan";
    else if (count < target) result.reason = "target_overshot";
    else if (count > target) result.reason = "target_not_reached";
    else if (result.violationSteps.length) result.reason = "forbidden_pair";
    else result.reason = result.finalMatches ? "target_reached" : "final_mismatch";
  }

  const currentSuccess = result.goalMet === true && result.violationSteps.length === 0;
  if (currentSuccess) result.feedback = "Correct";
  else if (result.violationSteps.length) {
    result.feedback = result.goalMet ? "Correct, but rule violation" : "Incorrect, also rule violation";
    result.feedback += ". Rule violation: move " + result.violationSteps[0] + ".";
  }
  if (hasPrevious) {
    result.eligible = result.completed && result.finalMatches === true;
    result.persistentError = result.eligible && !previousViolation && !currentSuccess;
    if (currentSuccess) result.ruleOutcome = previousViolation ? "current_only" : "both";
    else if (!result.eligible) result.ruleOutcome = "ineligible";
    else result.ruleOutcome = result.persistentError ? "persistent_error" : "general_error";
  }
  return result;
}
