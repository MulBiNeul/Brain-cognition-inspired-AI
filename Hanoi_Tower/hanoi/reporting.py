"""Read, replay, and export experiment logs for evaluators, independently of API requests."""
import csv
from html import escape
import io
import json
import os
from pathlib import Path
import tempfile
from .terminal import candidate_rule_text, condition_text, describe_moves, initial_stacks, initial_state_text, rule_text
from .prompts import feedback_values
_UNSPECIFIED_GOAL = object()

def _candidate_protocol(frozen):
    return isinstance(frozen, dict) and frozen.get('version') in (24, 25, 26, 27) and (frozen.get('dataset_version') == 7)

def _candidate_trace(moves, rule, peg_names, disk_count, goal_peg, initial_positions):
    """Replay the saved candidate rule; forbidden basic-legal moves still execute."""
    from .rules import replay_plan as replay_candidate
    result = {'success': False, 'reason': '', 'failed_at': None, 'steps': [], 'goal_peg': goal_peg}
    stacks = initial_stacks(initial_positions, pegs=peg_names, disk_count=disk_count)
    if stacks is None or disk_count != 3 or tuple(peg_names) != ('A', 'B', 'C', 'D'):
        result['reason'] = '퍼즐 배치 기록 형식 오류'
        return result
    result['steps'].append({'index': 0, 'move': None, 'disk': None, 'valid': True, 'base_valid': True, 'rule_violation': False, 'message': '시작 배치', 'state': stacks})
    if goal_peg not in peg_names:
        result['reason'] = '목표 기둥 기록 형식 오류'
        return result
    if not isinstance(moves, (list, tuple)) or not 1 <= len(moves) <= 30:
        result['reason'] = '계획 형식 오류: 1~30개의 이동 명령이 필요합니다.'
        return result
    try:
        trace = replay_candidate(initial_positions, goal_peg, moves, rule['candidate_rule'])
    except (KeyError, TypeError, ValueError):
        result['reason'] = '규칙 또는 초기 배치 기록 형식 오류'
        return result
    for step in trace['steps']:
        violates = step['rule_violation']
        message = '숨은 규칙 위반 (진단 재생 계속): ' + candidate_rule_text(rule['candidate_rule']) if violates else '이동 성공'
        result['steps'].append({'index': step['step'], 'move': step['move'], 'disk': step['disk'], 'valid': not violates, 'base_valid': True, 'rule_violation': violates, 'message': message, 'state': initial_stacks(step['positions'], pegs=peg_names)})
        if violates and result['failed_at'] is None:
            result['failed_at'] = step['step']
    error = trace['basic_error']
    if error:
        before = result['steps'][-1]['state']
        move = error['move']
        disk = before[move[0]][-1] if isinstance(move, str) and len(move) == 2 and (move[0] in before) and before[move[0]] else None
        result['steps'].append({'index': error['step'], 'move': move, 'disk': disk, 'valid': False, 'base_valid': False, 'rule_violation': None, 'message': '기본 하노이 이동 오류', 'state': before})
        result['failed_at'] = result['failed_at'] or error['step']
    result['success'] = trace['base_legal'] and trace['rule_compliant'] and trace['goal_reached']
    if error:
        result['reason'] = '기본 하노이 이동 오류'
    elif not trace['rule_compliant']:
        result['reason'] = f"숨은 규칙 위반: {len(trace['rule_violation_steps'])}회 (기본 이동은 계속 재생)"
    elif result['success']:
        result['reason'] = f'목표 {goal_peg} 달성'
    else:
        result['reason'] = f'모든 이동은 유효하지만 최종 배치가 목표 {goal_peg}에 도달하지 않았습니다.'
    return result

def pe_definition(version):
    return '지속적 오류 = 현재 라운드의 추가 이동 제한을 하나 이상 위반한 제출. 위반한 제한이 여러 개여도 제출당 1회로 계산합니다. 목적지 불일치는 별도 오류이며 PE에 포함하지 않습니다. 직전 라운드의 성공 여부는 PE와 무관합니다.'

def movement_diagnostics(record):
    """Show saved protocol diagnostics without regrading historical records under new definitions."""
    if 'movement_rule_error' not in record:
        return ''
    violations = record.get('movement_rule_violations') or []
    goal = {True: '목적지 불일치', False: '목적지 도달', None: '목적지 판정 불가 (기본 이동 오류)'}.get(record.get('goal_error'))
    details = [f"{item.get('step')}번째 {item.get('move')} · 원판 {(item.get('disk') if item.get('disk') is not None else '확인 불가')}" for item in violations]
    return f'추가 이동 제한 위반 {len(violations)}건 · {goal}' + (' · ' + '; '.join(details) if details else '')

def replay_plan(moves, rule_dict_or_None=None, *, peg_names=None, disk_count=3, goal_peg=_UNSPECIFIED_GOAL, initial_positions=None):
    """Replay a supported protocol from its explicit saved state."""
    rule = rule_dict_or_None
    if not isinstance(rule, dict) or not isinstance(rule.get('candidate_rule'), dict):
        return {'success': False, 'reason': '규칙 기록 형식 오류', 'failed_at': None, 'steps': [], 'goal_peg': None}
    if goal_peg is _UNSPECIFIED_GOAL:
        goal_peg = rule.get('goal_peg')
    if peg_names is None:
        peg_names = ('A', 'B', 'C', 'D')
    if not isinstance(peg_names, (list, tuple)) or tuple(peg_names) != ('A', 'B', 'C', 'D') or type(disk_count) is not int or (disk_count != 3):
        return {'success': False, 'reason': '퍼즐 배치 기록 형식 오류', 'failed_at': None, 'steps': [], 'goal_peg': goal_peg}
    if initial_positions is None:
        initial_positions = rule.get('initial_positions')
    return _candidate_trace(moves, rule, peg_names, disk_count, goal_peg, initial_positions)

def _json(path, errors, default=None):
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('JSON 객체가 필요합니다')
        return data
    except (OSError, ValueError) as exc:
        errors.append(f'{path.name}: {type(exc).__name__} — 기록을 읽을 수 없습니다.')
        return default

def _jsonl(path, errors):
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError):
        errors.append(f'{path.name}: 기록을 읽을 수 없습니다.')
        return []
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('JSON 객체가 필요합니다')
            rows.append(row)
        except ValueError:
            errors.append(f'{path.name}:{number}: 손상된 JSON 기록입니다.')
    return rows

def round_policy(frozen, number):
    policies = frozen.get('round_policies', [])
    if isinstance(policies, list) and type(number) is int and (1 <= number <= len(policies)):
        policy = policies[number - 1]
        return policy if isinstance(policy, dict) else {}
    return {}

def problem_rules(frozen, episode):
    problems = frozen.get('problems', [])
    if isinstance(problems, list) and type(episode) is int and (1 <= episode <= len(problems)):
        problem = problems[episode - 1]
        rounds = problem.get('rounds', []) if isinstance(problem, dict) else []
        return rounds if isinstance(rounds, list) else []
    return []

def problem_condition(frozen, episode):
    problems = frozen.get('problems', [])
    if isinstance(problems, list) and type(episode) is int and (1 <= episode <= len(problems)):
        problem = problems[episode - 1]
        return problem.get('condition') if isinstance(problem, dict) else None
    return None

def _early_rounds_complete(frozen, config, summary, records):
    """Accept early completion only when both submissions and round summaries confirm it."""
    episodes = config.get('episodes')
    if type(episodes) is not int or episodes < 1:
        return False
    episode_rows = summary.get('episodes', [])
    if len(episode_rows) != episodes or [row.get('episode') for row in episode_rows] != list(range(1, episodes + 1)):
        return False
    resolved = summary.get('completed_rounds', 0) + summary.get('excluded_rounds', 0)
    if resolved != episodes * 5 or summary.get('total_rounds') != episodes * 5:
        return False
    if any((type(record.get('current_success')) is not bool for record in records)):
        return False
    expected_positions = []
    for episode in episode_rows:
        rounds = episode.get('rounds', [])
        if episode.get('complete') is not True or [row.get('round') for row in rounds] != list(range(1, 6)):
            return False
        exhausted = None
        for row in rounds:
            if row.get('status') == 'excluded':
                if exhausted is None or row.get('attempts') != 0 or row.get('complete') is not False or (row.get('excluded_reason') != 'prior_round_exhausted') or (row.get('excluded_after_round') != exhausted):
                    return False
                if any((record.get('episode') == episode['episode'] and record.get('round') == row.get('round') for record in records)):
                    return False
                continue
            if exhausted is not None:
                return False
            policy = round_policy(frozen, row.get('round'))
            if policy.get('round') != row.get('round'):
                return False
            limit, required_streak = (policy.get('max_attempts'), policy.get('success_streak_required'))
            if type(limit) is not int or limit < 1 or type(required_streak) is not int or required_streak < 1:
                return False
            if required_streak > limit:
                return False
            actual = [record for record in records if record.get('episode') == episode['episode'] and record.get('round') == row['round']]
            if not actual or len(actual) > limit:
                return False
            streak = peak = 0
            first_success = criterion_met_at = None
            for index, record in enumerate(actual, 1):
                streak = streak + 1 if record['current_success'] else 0
                peak = max(peak, streak)
                if record['current_success'] and first_success is None:
                    first_success = index
                if streak >= required_streak and criterion_met_at is None:
                    criterion_met_at = index
                if record.get('success_streak') != streak or record.get('round_success') is not (streak >= required_streak):
                    return False
            if criterion_met_at is not None and criterion_met_at != len(actual):
                return False
            expected_status = 'success' if criterion_met_at is not None else 'exhausted' if len(actual) == limit else None
            if expected_status is None or row.get('status') != expected_status or row.get('complete') is not True:
                return False
            if row.get('attempts') != len(actual) or row.get('max_attempts') != limit:
                return False
            if row.get('first_success_attempt') != first_success:
                return False
            if row.get('success_streak_required') != required_streak or row.get('success_streak') != streak or row.get('max_success_streak') != peak or (row.get('criterion_met_at') != criterion_met_at):
                return False
            if expected_status == 'exhausted':
                exhausted = row['round']
            expected_positions.extend(((episode['episode'], row['round'], attempt) for attempt in range(1, len(actual) + 1)))
    if summary.get('completed_rounds') != sum((row.get('complete') is True for episode in episode_rows for row in episode['rounds'])) or summary.get('excluded_rounds') != sum((row.get('status') == 'excluded' for episode in episode_rows for row in episode['rounds'])):
        return False
    return [(record.get('episode'), record.get('round'), record.get('attempt')) for record in records] == expected_positions

def snapshot(directory):
    """Read per-file atomic checkpoints and expose cross-file inconsistencies during writes."""
    directory = Path(directory)
    errors, warnings = ([], [])
    protocol = _json(directory / 'protocol.json', errors, {})
    frozen = protocol.get('frozen', protocol)
    if not isinstance(frozen, dict):
        errors.append('protocol.json: frozen 객체 형식 오류')
        frozen = {}
    config = frozen.get('config', {})
    if not isinstance(config, dict):
        errors.append('protocol.json: config 객체 형식 오류')
        config = {}
    if not _candidate_protocol(frozen):
        errors.append('지원하지 않는 프로토콜입니다. 이전 실험은 보관된 뷰어에서 확인하세요.')
    allowed_goals = ('A', 'B', 'C', 'D')
    count = config.get('episodes', 1)
    if type(count) is not int or not 1 <= count <= 300:
        errors.append('protocol.json: 문제 수 기록 형식 오류')
    else:
        for episode in range(1, count + 1):
            rounds = problem_rules(frozen, episode)
            if not isinstance(rounds, list) or len(rounds) != 5 or any((not isinstance(rule, dict) or rule.get('goal_peg') not in allowed_goals for rule in rounds)):
                errors.append(f'protocol.json: 문제 {episode}의 라운드별 목표 기둥 기록 형식 오류')
            from .rules import replay_plan as replay_candidate
            for number, rule in enumerate(rounds, 1):
                try:
                    if not isinstance(rule, dict) or not isinstance(rule.get('candidate_rule'), dict):
                        raise ValueError('candidate rule missing')
                    replay_candidate(rule.get('initial_positions'), rule.get('goal_peg'), [], rule['candidate_rule'])
                except (KeyError, TypeError, ValueError):
                    errors.append(f'protocol.json: 문제 {episode} R{number}의 초기 배치·추가 제한 기록 형식 오류')
    policies = frozen.get('round_policies')
    if not isinstance(policies, list) or len(policies) != 5 or any((not isinstance(policy, dict) or policy.get('round') != number or policy.get('max_attempts') != config.get('max_attempts') or (policy.get('success_streak_required') != (config.get('success_streak') if number > 1 or config.get('r1_apply_k') else 1)) for number, policy in enumerate(policies, 1))):
        errors.append('protocol.json: 라운드별 종료 기준 기록 형식 오류')
    problems = frozen.get('problems', [])
    allowed_conditions = ('peg', 'occupancy', 'source_sequence', 'visit')
    if not isinstance(problems, list) or len(problems) != config.get('episodes') or any((not isinstance(problem, dict) or problem.get('condition') not in allowed_conditions for problem in problems)):
        errors.append('protocol.json: 문제별 규칙 유형 기록 형식 오류')
    else:
        for problem in problems:
            rounds = problem.get('rounds')
            for rule in rounds if isinstance(rounds, list) else []:
                if isinstance(rule, dict) and isinstance(rule.get('candidate_rule'), dict) and (rule['candidate_rule'].get('kind') != problem.get('condition')):
                    errors.append(f"protocol.json: {problem.get('id')} 문제 안의 규칙 유형 불일치")
    summary = _json(directory / 'summary.json', errors, {})
    if not isinstance(summary.get('pooled', {}), dict) or not isinstance(summary.get('episodes', []), list):
        errors.append('summary.json: pooled 또는 episodes 형식 오류')
        summary = {}
    elif any((not isinstance(episode, dict) or not isinstance(episode.get('rounds', []), list) or any((not isinstance(row, dict) for row in episode.get('rounds', []))) for episode in summary.get('episodes', []))):
        errors.append('summary.json: 라운드 요약 형식 오류')
        summary = {}
    records = _jsonl(directory / 'submissions.jsonl', errors)
    if not (directory / 'submissions.jsonl').exists():
        for file in sorted((directory / 'trials').glob('*/record.json')):
            record = _json(file, errors)
            if record is not None:
                records.append(record)
        if records:
            warnings.append('submissions.jsonl이 없어 개별 record.json에서 기록을 읽었습니다.')
    required = ('call', 'episode', 'round', 'attempt', 'current_success', 'previous_success', 'persistent_error')
    if any((any((key not in record for key in required)) for record in records)):
        errors.append('시도 기록에 필수 채점 필드가 없습니다.')
    calls = [record.get('call') for record in records]
    failure_records = [_json(file, errors, {}) for file in sorted((directory / 'trials').glob('*/generation_failure.json'))]
    all_calls = sorted(calls + [row.get('call', 0) for row in failure_records])
    if calls != sorted(set(calls)) or all_calls != list(range(1, len(all_calls) + 1)):
        errors.append('시도 기록의 호출 번호가 중복되었거나 순서가 연속적이지 않습니다.')
    events = _jsonl(directory / 'events.jsonl', errors)
    stop_info = _json(directory / 'last_stop.json', errors)
    state = _json(directory / 'state.json', errors, {})
    if not state:
        status = 'complete' if summary.get('complete') is True else 'preview' if (directory / 'first_request_preview.json').exists() else 'ready'
        state = {'phase': status}
    if not isinstance(state.get('phase'), str):
        errors.append('state.json: phase 필드가 없거나 형식이 올바르지 않습니다.')
    count = summary.get('recorded_submissions')
    if count is not None and count != len(records):
        warnings.append('요약과 시도 기록 수가 다릅니다. 저장 중이거나 checkpoint를 확인해야 합니다.')
        if state.get('phase') == 'complete':
            state = {**state, 'phase': 'incomplete'}
    complete_evidence = _early_rounds_complete(frozen, config, summary, records)
    if state.get('phase') == 'complete' and (not records or not complete_evidence or summary.get('complete') is not True):
        warnings.append('완료 표시에 대응하는 전체 채점 기록이 없어 미완료로 표시합니다.')
        state = {**state, 'phase': 'incomplete'}
    if errors:
        state = {**state, 'phase': 'error'}
    try:
        console = (directory / 'console.log').read_text(encoding='utf-8', errors='replace').splitlines()[-200:]
    except FileNotFoundError:
        console = []
    except OSError:
        console = []
        warnings.append('console.log를 읽을 수 없습니다.')
    return {'name': directory.name, 'protocol': protocol, 'config': config, 'state': state, 'summary': summary, 'records': records, 'generation_failures': failure_records, 'events': events, 'console_tail': '\n'.join(console), 'stop_info': stop_info, 'errors': errors, 'warnings': warnings}

def trial_detail(directory, call):
    directory, errors = (Path(directory), [])
    if isinstance(call, bool) or not isinstance(call, int) or call < 1:
        raise ValueError('call은 양의 정수여야 합니다.')
    trials = sorted((directory / 'trials').glob(f'{call:05d}_*'))
    if len(trials) != 1:
        raise FileNotFoundError(f'호출 {call}의 시도 폴더를 하나로 식별할 수 없습니다.')
    trial = trials[0]
    record = _json(trial / 'record.json', errors, {})
    request = _json(trial / 'request.json', errors)
    responses = sorted(trial.glob('transport_*_response.json'))
    response = None
    if len(responses) == 1:
        envelope = _json(responses[0], errors, {})
        response = envelope.get('response')
    elif len(responses) > 1:
        errors.append('한 시도에 API 응답이 여러 개 있어 원본을 선택하지 않았습니다.')
    protocol = _json(directory / 'protocol.json', errors, {})
    frozen = protocol.get('frozen', protocol)
    rules = problem_rules(frozen, record.get('episode')) if isinstance(frozen, dict) else []
    allowed_goals = ('A', 'B', 'C', 'D')
    peg_names = frozen.get('pegs', ())
    disk_count = frozen.get('disk_count', 3) if isinstance(frozen, dict) else 3
    number = record.get('round')
    current_trace = previous_trace = None
    if not _candidate_protocol(frozen):
        errors.append('지원하지 않는 프로토콜 버전으로 이동을 재생할 수 없습니다.')
    elif isinstance(number, int) and 1 <= number <= len(rules):
        current_rule = rules[number - 1]
        current_goal = current_rule.get('goal_peg') if isinstance(current_rule, dict) else None
        if current_goal not in allowed_goals:
            errors.append('현재 라운드의 목표 기둥 기록 형식 오류')
        current_trace = replay_plan(record.get('moves'), current_rule, peg_names=peg_names, disk_count=disk_count, goal_peg=current_goal)
        if number > 1:
            previous_rule = rules[number - 2]
            previous_goal = previous_rule.get('goal_peg') if isinstance(previous_rule, dict) else None
            if isinstance(previous_rule, dict) and isinstance(current_rule, dict):
                previous_rule = {**previous_rule, 'initial_positions': current_rule.get('initial_positions'), 'goal_peg': current_goal}
                previous_goal = current_goal
            if previous_goal not in allowed_goals:
                errors.append('직전 라운드의 목표 기둥 기록 형식 오류')
            previous_trace = replay_plan(record.get('moves'), previous_rule, peg_names=peg_names, disk_count=disk_count, goal_peg=previous_goal)
    elif record:
        errors.append('라운드 규칙 기록이 없어 이동을 재생할 수 없습니다.')
    return {'record': record, 'generation_failure': _json(trial / 'generation_failure.json', errors), 'request': request, 'response': response, 'current_trace': current_trace, 'previous_trace': previous_trace, 'errors': errors}

def csv_text(data):
    fields = ['protocol_version', 'provider', 'model', 'episode', 'problem_id', 'condition', 'condition_label', 'round', 'attempt', 'feedback_policy', 'feedback_code', 'feedback', 'correctness_feedback', 'current_success', 'previous_success', 'persistent_error', 'movement_rule_error', 'movement_rule_violations', 'goal_error', 'goal_reached', 'base_valid', 'base_error', 'success_streak', 'round_success', 'success_streak_required', 'move_count', 'moves', 'elapsed_seconds', 'response_id']
    protocol = data.get('protocol', {})
    frozen = protocol.get('frozen', protocol)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in data.get('records', []):
        row = {key: record.get(key) for key in fields}
        row['protocol_version'] = frozen.get('version')
        config = frozen.get('config', {})
        row['feedback_policy'] = record.get('feedback_policy', frozen.get('feedback_policy', config.get('feedback_policy', 'binary_v1')))
        row['provider'] = config.get('provider', 'openai')
        row['model'] = config.get('model')
        row['condition'] = problem_condition(frozen, record.get('episode'))
        row['condition_label'] = condition_text(row['condition'])
        row['success_streak_required'] = round_policy(frozen, record.get('round')).get('success_streak_required')
        moves = record.get('moves')
        row.update(move_count=len(moves) if isinstance(moves, list) else 0, moves=' '.join((str(move) for move in moves)) if isinstance(moves, list) else '')
        for key in ('movement_rule_violations', 'base_error'):
            row[key] = json.dumps(record[key], ensure_ascii=False) if key in record else None
        for key, value in row.items():
            if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
                row[key] = "'" + value
        writer.writerow(row)
    return '\ufeff' + stream.getvalue()

def _text(value):
    return escape('—' if value is None else str(value), quote=True)

def _pretty(value):
    return escape(json.dumps(value, ensure_ascii=False, indent=2), quote=True)

def _rate(value):
    return f'{value:.1%}' if isinstance(value, (float, int)) and (not isinstance(value, bool)) else '—'

def html_text(data):
    summary, config, records = (data.get('summary', {}), data.get('config', {}), data.get('records', []))
    frozen = data.get('protocol', {}).get('frozen', data.get('protocol', {}))
    feedback_policy = frozen.get('feedback_policy', config.get('feedback_policy', 'binary_v1'))
    supplied_feedback_values = frozen.get('feedback_values')
    if not isinstance(supplied_feedback_values, list) or not supplied_feedback_values or not all(isinstance(value, str) for value in supplied_feedback_values):
        try:
            supplied_feedback_values = feedback_values(feedback_policy)
        except ValueError:
            supplied_feedback_values = ('피드백 정책 기록 확인 필요',)
    feedback_labels = ' / '.join(f'“{_text(value)}”' for value in supplied_feedback_values)
    pooled = summary.get('pooled') or {}
    error_rounds = 'R1–R5'
    error_problem_count_field = 'error_metric_eligible_problems'
    error_round_count_field = 'eligible_error_rounds'
    error_problem_scope = '실제 채점 제출이 있는'
    primary_metric = f"<div class='card'>Total MPE<b>{_rate(summary.get('total_mpe'))}</b>R2 이상 진행된 {_text(summary.get('metric_eligible_problems'))}문제의 MPE 평균</div>"
    error_metric = f"<div class='card'>전체 평균 오답률 · {error_rounds}<b>{_rate(summary.get('total_mean_error_rate'))}</b>{error_problem_scope} {_text(summary.get(error_problem_count_field))}문제의 평균 오답률</div>"
    metric_method = '라운드 PER = 해당 라운드의 지속적 오류 제출 수 / 실제 채점 제출 수. 문제별 MPE = 진행된 R2–R5의 PER 합 / 해당 라운드 수. Total MPE 및 유형별 MPE = 해당 문제들의 MPE 합 / R2 이상 진행된 문제 수. 채점 제출이 없는 라운드는 제외하며, 진행 중이거나 한도 종료된 라운드의 실제 제출은 포함합니다.'
    metric_method += f' 오답률도 같은 순서로 평균합니다. 라운드 오답률 = 오답 제출 수 / 실제 채점 제출 수. 문제별 평균 오답률 = 진행된 {error_rounds} 오답률의 산술평균. 전체 및 유형별 평균 오답률 = {error_problem_scope} 문제들의 평균 오답률의 산술평균. 참고용 제출 통합 오답률(pooled)은 R2–R5의 전체 오답 수 / 전체 제출 수로 별도 표시합니다.'
    status = data.get('state', {}).get('phase', 'ready')
    labels = {'complete': '완료', 'running': '실행 중', 'ready': '준비', 'preview': '요청 미리보기', 'finished_with_failures': '실행 종료 · 미완료 문제 있음', 'incomplete': '미완료', 'error': '오류', 'stopped': '중단', 'stopping': '중단 중'}
    rounds = []
    rule_data = frozen.get('rounds', []) if isinstance(frozen, dict) else []
    round_labels = {'success': '연속 정답 기준 달성', 'exhausted': '시도 한도 소진', 'pending': '대기', 'in_progress': '진행 중', 'generation_failed': '생성 중단 · 미채점', 'excluded': '미실행 · 제외'}
    for episode in summary.get('episodes', []):
        episode_rules = problem_rules(frozen, episode.get('episode'))
        for row in episode.get('rounds', []):
            number = row.get('round')
            required_streak = round_policy(frozen, number).get('success_streak_required')
            label = rule_text(episode_rules[number - 1], require_goal=True) if type(number) is int and 1 <= number <= len(episode_rules) else '규칙 정보 없음'
            if type(number) is int and 1 <= number <= len(episode_rules):
                label = '시작 ' + initial_state_text(episode_rules[number - 1]) + ' · ' + label
            values = [episode.get('problem_id', episode.get('episode')), condition_text(problem_condition(frozen, episode.get('episode'))), number, label, row.get('attempts'), row.get('successes'), row.get('failures'), _rate(row.get('failure_rate')), row.get('persistent_errors'), _rate(row.get('persistent_error_rate')), row.get('first_success_attempt'), row.get('criterion_met_at', row.get('first_success_attempt') if required_streak == 1 else None), f"{row.get('success_streak', 1 if row.get('status') == 'success' else 0)}/{required_streak}", row.get('max_success_streak'), round_labels.get(row.get('status'), '고정 횟수')]
            rounds.append('<tr>' + ''.join((f'<td>{_text(value)}</td>' for value in values)) + '</tr>')
    attempts = []
    for record in records:
        moves = record.get('moves')
        plan = ' → '.join(map(str, moves)) if isinstance(moves, list) else '형식 오류'
        rules = problem_rules(frozen, record.get('episode'))
        number = record.get('round')
        if type(number) is int and 1 <= number <= len(rules) and isinstance(rules[number - 1], dict):
            positions = rules[number - 1].get('initial_positions')
            plan = ' · '.join(describe_moves(moves, initial_positions=positions) or ['형식 오류'])
        label = f"#{record.get('call')} · {record.get('problem_id', 'E' + str(record.get('episode')))} · {condition_text(problem_condition(frozen, record.get('episode')))} · R{record.get('round')}-{record.get('attempt')} · {record.get('feedback')}"
        checks = {key: record.get(key) for key in ('feedback_policy', 'feedback_code', 'feedback', 'correctness_feedback', 'current_success', 'previous_success', 'persistent_error', 'movement_rule_error', 'movement_rule_violations', 'goal_error', 'goal_reached', 'base_valid', 'base_error', 'success_streak', 'round_success', 'parse_valid', 'elapsed_seconds', 'response_id', 'usage', 'reasoning_replay') if key in record}
        diagnostic = f'<p>{_text(movement_diagnostics(record))}</p>'
        attempts.append(f"<details><summary>{_text(label)} · 지속적 오류: {_text(record.get('persistent_error'))}</summary><p class='plan'>{_text(plan)}</p>{diagnostic}<h3>채점 기록</h3><pre>{_pretty(checks)}</pre><h3>모델의 실제 출력</h3><pre>{_text(record.get('raw_output'))}</pre></details>")
    diagnostics = ''.join((f'<li>{_text(message)}</li>' for message in data.get('errors', []) + data.get('warnings', [])))
    rule_data = frozen.get('problems', [])
    r1 = round_policy(frozen, 1)
    later = round_policy(frozen, 2)
    progress = f"완료 라운드 {_text(summary.get('completed_rounds'))} / {_text(summary.get('total_rounds'))} · 전체 최대 {_text(summary.get('max_submissions'))}회 제출"
    method = f"R1: T={r1.get('max_attempts')}, 연속 정답 {r1.get('success_streak_required')}회. R2–R5: T={later.get('max_attempts')}, k={later.get('success_streak_required')}. 기준 달성 시 다음 라운드로 진행합니다. T 내 기준 미달 시 해당 문제의 남은 라운드를 미실행 처리하고 다음 문제로 진행합니다. R1은 PER 및 MPE에서 제외합니다. 라운드 PER의 분모는 실제 채점 제출 수입니다. 문제 간 대화는 초기화합니다."
    progress += f" · 미실행 {_text(summary.get('excluded_rounds', 0))}라운드"
    method += ' 한 문제 안에서는 같은 규칙 유형을 유지하고, R1–R5의 초기 배치·목표·추가 제한을 변경합니다. 제한은 누적하지 않고 라운드마다 교체하며, 각 제출은 해당 라운드의 초기 배치와 초기 규칙 이력에서 시작합니다. 직전 규칙 비교에는 현재 초기 배치와 목표를 사용합니다.'
    method += ' 출력 방식: ' + ('일반 텍스트로 JSON 요청 · API 스키마 강제 없음 · 로컬 공통 채점.' if config.get('output_format') == 'plain' else 'API 스키마 강제 · 별도 진단 설정.')
    method += ' 추론 상태는 같은 문제의 R1–R5에서 유지하고 문제 변경 시 초기화합니다. Sol 암호화 reasoning 항목과 Opus 원본 signed thinking 블록을 전달합니다.'
    method += f' 모델 피드백은 {feedback_labels} 중 하나입니다.'
    if feedback_policy == 'goal_rule_4way_v1':
        method += ' 정답 / 오답 / 규칙 위반 / 정답이지만 규칙 위반을 구분합니다. “Correct, but rule violation”은 목표 배치에 도달했지만 이동 제한을 위반한 오답이며, 연속 정답 수를 0으로 초기화합니다.'
    peg_names = frozen.get('pegs', [])
    destination = '목표는 라운드별 변경 · 모델에게 비공개'
    start_label = '초기 배치는 라운드별 변경'
    puzzle = f"원판 {_text(frozen.get('disk_count', 3))}개 · 기둥 {len(peg_names)}개 ({_text(' · '.join(peg_names))}) · {start_label} → {destination}" if isinstance(peg_names, (list, tuple)) and all((isinstance(peg, str) for peg in peg_names)) else '퍼즐 배치 기록 형식 오류'
    conditions = summary.get('conditions', {})
    condition_rows = ''.join(('<tr>' + ''.join((f'<td>{_text(value)}</td>' for value in (item.get('label', condition_text(key)), item.get('problem_count'), item.get('metric_eligible_problems'), _rate(item.get('mpe')), *[item.get(error_problem_count_field), _rate(item.get('mean_error_rate'))], item.get('attempts'), item.get('persistent_errors'), _rate(item.get('persistent_error_rate'))))) + '</tr>' for key, item in conditions.items() if isinstance(item, dict))) if isinstance(conditions, dict) else ''
    condition_table = f'<h2>조건별 결과 · MPE R2–R5 / 오답률 {error_rounds}</h2>' + '<table><thead><tr><th>조건</th><th>계획 문제</th>' + '<th>MPE 문제 N</th><th>MPE</th>' + f'<th>오답률 문제 N</th><th>평균 오답률 · {error_rounds}</th>' + '<th>R2–R5 제출</th><th>R2–R5 지속적 오류</th><th>참고 PER (pooled)</th></tr></thead><tbody>' + condition_rows + '</tbody></table>' if condition_rows else ''
    problem_rows = ''.join(('<tr>' + ''.join((f'<td>{_text(value)}</td>' for value in (item.get('problem_id', item.get('episode')), *(_rate((item.get('per_by_round') or {}).get(str(number))) for number in range(2, 6)), item.get('eligible_scored_rounds'), _rate(item.get('mpe')), *[item.get(error_round_count_field), _rate(item.get('mean_error_rate'))]))) + '</tr>' for item in summary.get('episodes', [])))
    problem_table = '<h2>문제별 MPE 및 평균 오답률</h2>' + '<table><thead><tr><th>문제</th><th>R2 PER</th><th>R3 PER</th><th>R4 PER</th><th>R5 PER</th><th>MPE 라운드 수</th><th>MPE</th>' + f'<th>오답률 라운드 수</th><th>평균 오답률 · {error_rounds}</th>' + '</tr></thead><tbody>' + problem_rows + '</tbody></table>' if problem_rows else ''
    goal_summary = ''
    scored = [record for record in records if record.get('round', 0) > 1]
    goal_errors = sum((record.get('goal_error') is True for record in scored))
    goal_unknown = sum((record.get('goal_error') is None for record in scored))
    goal_summary = f"<p class='muted'>목적지 불일치: {goal_errors}회 · 목적지 판정 불가: {goal_unknown}회 (R2–R5). 목적지 오류와 이동 제한 위반은 동시에 발생할 수 있어 합산하지 않습니다.</p>"
    failure_panel = ''
    if data.get('generation_failures'):
        failure_panel = '<h2>생성 중단 · 미채점</h2><p>자동 재시도 없이 해당 문제의 남은 라운드를 제외했습니다. 이 응답은 오류율의 분자·분모에 포함하지 않습니다. 조기 수신 종료 시 토큰 사용량은 일부만 관측되어 실제 비용보다 적을 수 있습니다.</p><pre>' + _pretty(data['generation_failures']) + '</pre>'
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>하노이탑 실험 기록 · {_text(data.get('name'))}</title><style>
body{{font:15px/1.6 system-ui,sans-serif;background:#f4f6fa;color:#192336;max-width:1150px;margin:32px auto;padding:0 20px}}h1{{font-size:28px}}h2{{margin-top:30px}}h3{{font-size:14px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card,details,.panel{{background:white;border:1px solid #dbe1eb;border-radius:10px;padding:16px}}.card{{flex:1;min-width:160px}}.card b{{display:block;font-size:23px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:9px;border-bottom:1px solid #dbe1eb;text-align:left}}.table-wrap{{overflow:auto}}details{{margin:10px 0}}summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f7;padding:12px;border-radius:6px;font-size:13px}}.plan{{font-family:monospace;overflow-wrap:anywhere}}.muted{{color:#5c6675}}.warning{{color:#943c10}}footer{{margin:36px 0;color:#5c6675}}
</style><h1>하노이탑 API 실험 기록</h1><p>{_text(data.get('name'))} · {_text(labels.get(status, status))}</p><p>{puzzle}</p>
<div class="cards"><div class="card">모델 / 추론 강도<b>{_text(config.get('model'))}</b>{_text(config.get('effort'))}</div>
<div class="card">실제 채점 제출<b>{len(records)}회</b>{progress}</div>
{primary_metric}
{error_metric}
<div class="card">참고: 제출 통합 오답률 (pooled)<b>{_rate(pooled.get('failure_rate'))}</b>R2–R5 오답 합 / 제출 합</div></div>
<p class="muted">{method} {pe_definition(frozen.get('version'))} R1은 MPE에서 제외하고 오답률에는 포함합니다. 미완료 실행의 값은 중간 집계입니다. 같은 실험 안의 제출은 서로 독립인 표본이 아닙니다.</p>
<p class="muted">{metric_method}</p>
{goal_summary}
{('<ul class="warning">' + diagnostics + '</ul>' if diagnostics else '')}
{condition_table}
{problem_table}
{failure_panel}
<h2>라운드별 결과</h2><div class="table-wrap"><table><thead><tr><th>문제</th><th>조건</th><th>라운드</th><th>목적지·이동 제한</th><th>제출</th><th>정답</th><th>오답</th><th>라운드 오답률</th><th>지속적 오류</th><th>PER</th><th>첫 정답</th><th>연속 기준 달성</th><th>최종 연속 / k</th><th>최대 연속</th><th>종료 상태</th></tr></thead><tbody>{''.join(rounds)}</tbody></table></div>
<h2>실제 제출 로그</h2><p class="muted">각 항목을 눌러 전체 계획과 실제 출력, 현재·직전 라운드 판정과 별도의 이동 제한·목적지 진단을 확인하세요.</p>{''.join(attempts) or '<p>아직 채점된 제출이 없습니다.</p>'}
<details><summary>평가자 전용 목적지·이동 제한</summary><pre>{_pretty(rule_data)}</pre></details>
<details><summary>중단 정보</summary><pre>{_pretty(data.get('stop_info'))}</pre></details>
<details><summary>실행 콘솔 (최근 200줄)</summary><pre>{_text(data.get('console_tail', ''))}</pre></details>
<footer>이 보고서는 저장된 API 실험 기록으로 생성되었습니다. 구체적인 이동 제한·목표 기둥·라운드 정보·이동 실패 위치와 지속적 오류 분류명은 평가자 전용이며 모델에는 전달되지 않습니다. 모델 피드백은 {feedback_labels}뿐입니다. 문맥 방식: {_text(config.get('context_format'))}.</footer></html>"""

def _atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def write_reports(directory):
    directory = Path(directory)
    data = snapshot(directory)
    _atomic_text(directory / 'summary.html', html_text(data))
    _atomic_text(directory / 'submissions.csv', csv_text(data))
    return data
