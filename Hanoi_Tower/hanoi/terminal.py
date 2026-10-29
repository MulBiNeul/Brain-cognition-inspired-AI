"""Researcher-facing terminal output, never included in model input.

Uses only the standard library and omits ANSI colors when redirected to a file.
Displays saved records without changing experiment decisions or API requests.
"""
from collections.abc import Mapping
import json
import math
import os
import re
import sys
import unicodedata
_ANSI = re.compile('\\x1b\\][^\\x07\\x1b]*(?:\\x07|\\x1b\\\\)|\\x1b\\[[0-?]*[ -/]*[@-~]|\\x1b[@-_]')
_COLORS = {'cyan': '36', 'green': '32', 'red': '31', 'yellow': '33', 'dim': '2', 'bold': '1'}

def clean(value):
    """Remove ANSI/control characters from external text and collapse it to one line."""
    value = _ANSI.sub('', str(value))
    return ''.join((' ' if char in '\n\r\t' else char for char in value if char in '\n\r\t' or unicodedata.category(char) not in {'Cc', 'Cf', 'Cs'}))

def display_width(value):
    return sum((0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in {'W', 'F'} else 1 for char in clean(value)))

def pad(value, width, right=False):
    value = clean(value)
    gap = ' ' * max(0, width - display_width(value))
    return gap + value if right else value + gap

def wrap(value, width):
    """Wrap by display width, including Korean glyphs, without truncating long paths."""
    value = clean(value)
    lines, line, used = ([], '', 0)
    for char in value:
        size = display_width(char)
        if line and used + size > width:
            lines.append(line)
            line, used = ('', 0)
        line += char
        used += size
    return lines + [line]

def _field(rule, name):
    return rule.get(name) if isinstance(rule, Mapping) else getattr(rule, name, None)

def condition_text(condition):
    return {'peg': '기둥 조건', 'occupancy': '기둥 점유 조건', 'source_sequence': '연속 출발 조건', 'visit': '선행 방문 조건'}.get(condition, '유형 정보 없음') if isinstance(condition, str) else '유형 정보 없음'

def initial_stacks(initial_positions=None, *, pegs=('A', 'B', 'C', 'D'), disk_count=3):
    """Convert positions ordered by disk 1, 2, 3 into bottom-to-top peg stacks."""
    positions = ('A',) * disk_count if initial_positions is None else initial_positions
    if not isinstance(positions, (list, tuple)) or len(positions) != disk_count or any((not isinstance(peg, str) or peg not in pegs for peg in positions)):
        return None
    return {peg: [disk for disk in range(disk_count, 0, -1) if positions[disk - 1] == peg] for peg in pegs}

def initial_state_text(rule, *, pegs=('A', 'B', 'C', 'D')):
    stacks = initial_stacks(_field(rule, 'initial_positions'), pegs=pegs)
    return '  '.join((f"{peg}=[{','.join(map(str, disks))}]" for peg, disks in stacks.items())) if stacks else '초기 배치 기록 오류'

def candidate_rule_text(rule):
    if not isinstance(rule, Mapping):
        return '추가 제한 기록 오류'
    kind = rule.get('kind')
    if kind == 'peg':
        return f"모든 원판의 {clean(rule.get('source'))} → {clean(rule.get('destination'))} 이동 금지"
    if kind == 'occupancy':
        return f"{clean(rule.get('occupied_peg'))}에 원판이 있으면 모든 원판의 {clean(rule.get('source'))} → {clean(rule.get('destination'))} 이동 금지"
    if kind == 'source_sequence':
        return f"직전 이동이 {clean(rule.get('previous_source'))}에서 출발했으면 다음 이동은 {clean(rule.get('source'))}에서 출발 금지"
    if kind == 'visit':
        return f"원판 {clean(rule.get('visitor_disk'))}이 {clean(rule.get('visited_peg'))}를 방문하기 전 원판 {clean(rule.get('locked_disk'))} 이동 금지 (초기 배치 방문 인정)"
    return '추가 제한 기록 오류'

def _restrictions(rule):
    return (candidate_rule_text(_field(rule, 'candidate_rule')),) if rule is not None else ()

def rule_text(rule, *, require_goal=True):
    goal = _field(rule, 'goal_peg')
    goal_label = f'목표 {goal}' if isinstance(goal, str) and goal in ('A', 'B', 'C', 'D') else '목표 기록 오류'
    return goal_label + ' · ' + (' · '.join(_restrictions(rule)) or '추가 제한 기록 오류')

def _rate(numerator, denominator):
    if numerator is None or not denominator:
        return '—'
    return f'{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)'

def describe_moves(moves, *, pegs=('A', 'B', 'C', 'D'), disk_count=3, initial_positions=None):
    """Display a submitted plan with disk numbers without changing input or grading records.

    Each submission starts from its round's initial board. Additional restrictions
    do not stop diagnostic replay. Do not infer disks after the first basic move error.
    """
    if not isinstance(moves, (list, tuple)) or not moves:
        return None
    stacks = initial_stacks(initial_positions, pegs=pegs, disk_count=disk_count)
    if stacks is None:
        return None
    stopped = False
    labels = []
    for step, move in enumerate(moves, 1):
        valid_command = isinstance(move, str) and len(move) == 2 and (move[0] in stacks) and (move[1] in stacks) and (move[0] != move[1])
        direction = '→'.join(move) if valid_command else clean(move)
        disk, status = ('?', '')
        if stopped:
            status = ' (미실행: 앞선 기본 이동 오류)'
        elif not valid_command:
            status = ' (실패: 잘못된 명령)'
            stopped = True
        else:
            source, target = move
            if not stacks[source]:
                status = ' (실패: 빈 출발 기둥)'
                stopped = True
            else:
                disk = stacks[source][-1]
                if stacks[target] and disk > stacks[target][-1]:
                    status = ' (실패: 작은 원판 위에 큰 원판)'
                    stopped = True
                else:
                    stacks[target].append(stacks[source].pop())
        labels.append(f'{step}) 원판 {disk} {direction}{status}')
    return labels

class TerminalReporter:
    """Display progress, rules, move plans, and persistent error rates for researchers."""

    def __init__(self, directory, config, rules, max_attempts, required_successes, *, problems=None, round_policies=None):
        self.directory = directory
        self.config = config
        self.rules = tuple(rules)
        self.problems = tuple(problems or ())
        self.round_policies = tuple(round_policies or ())
        self.default_max_attempts = max_attempts
        self.default_required_successes = required_successes
        self.pegs = ('A', 'B', 'C', 'D')
        self.max_attempts = max_attempts
        self.required_successes = required_successes
        self.width = 88
        self.episode = None
        self.round_number = None
        self.streak = 0
        self.use_color = bool(getattr(sys.stdout, 'isatty', lambda: False)()) and 'NO_COLOR' not in os.environ and (os.environ.get('TERM') != 'dumb')

    def _policy(self, number):
        if self.round_policies:
            policy = self.round_policies[number - 1]
            return (_field(policy, 'max_attempts'), _field(policy, 'success_streak_required'))
        return (self.default_max_attempts, self.default_required_successes)

    def _problem_id(self, episode):
        return _field(self.problems[episode - 1], 'id') if self.problems else f'E{episode}'

    def _print(self, value=''):
        print(value, flush=True)

    def _color(self, value, color):
        value = clean(value)
        return f'\x1b[{_COLORS[color]}m{value}\x1b[0m' if self.use_color else value

    def _heading(self, title):
        self._print()
        title = f' {clean(title)} '
        self._print(self._color('╭' + title + '─' * max(0, self.width - 2 - display_width(title)) + '╮', 'cyan'))

    def _line(self, text='', color=None):
        inner = self.width - 4
        for line in wrap(text, inner):
            padded = pad(line, inner)
            self._print('│ ' + (self._color(padded, color) if color else padded) + ' │')

    def _close(self):
        self._print(self._color('╰' + '─' * (self.width - 2) + '╯', 'cyan'))

    def _detail(self, label, value, color=None):
        """Align labels and value columns even when a value wraps across lines."""
        label_width = 12
        for index, line in enumerate(wrap(value, self.width - 4 - label_width)):
            self._line(pad(label if index == 0 else '', label_width) + line, color)

    def _policy_groups(self):
        """Group consecutive rounds only when their effective stopping policies match."""
        groups = []
        for number in range(1, len(self.rules) + 1):
            policy = self._policy(number)
            if groups and groups[-1][2] == policy:
                groups[-1] = (groups[-1][0], number, policy)
            else:
                groups.append((number, number, policy))
        return groups

    def start(self):
        episodes = self.config.get('episodes', 1)
        policies = [self._policy(number) for number in range(1, len(self.rules) + 1)]
        maximum = episodes * sum(limit for limit, _ in policies)
        fourway = self.config.get('feedback_policy', 'binary_v1') == 'goal_rule_4way_v1'
        self._heading('하노이탑 · 네 가지 피드백 실험' if fourway else '하노이탑 · 정오답 피드백 실험')
        self._detail('모델', self.config.get('model', '—'), 'bold')
        self._detail('API / 추론', f"{self.config.get('provider', 'openai')} / {self.config.get('effort', '—')}")
        self._detail('실험 규모', f"{('서로 다른 문제' if self.problems else '독립 실험')} {episodes}개 × {len(self.rules)}라운드")
        if self.problems:
            self._detail('문제 번호', f'진행 순번 1–{episodes} · 각 문제의 원본 ID 유지 · 순번은 이번 실행 기준')
        conditions = {key: sum((_field(problem, 'condition') == key for problem in self.problems)) for key in ('peg', 'occupancy', 'source_sequence', 'visit')}
        condition_labels = [f'{condition_text(key)} {count}문제' for key, count in conditions.items() if count]
        for index in range(0, len(condition_labels), 2):
            self._detail('유형' if index == 0 else '', ' · '.join(condition_labels[index:index + 2]))
        start = '초기 배치는 라운드별 변경'
        self._detail('퍼즐 구성', f"원판 3개 · 기둥 {len(self.pegs)}개 ({'·'.join(self.pegs)})")
        self._detail('상태 / 목표', f'{start} · 목표는 라운드별 변경')
        self._line()
        self._line('진행 기준', 'bold')
        for first, last, (limit, required) in self._policy_groups():
            scope = f'R{first}' if first == last else f'R{first}–R{last} 공통'
            self._detail('적용 라운드', scope)
            self._detail('제출 한도', f'T = {limit}회 (라운드당 최대 제출)')
            self._detail('성공 기준', f'k = {required}회 연속 정답')
        self._detail('기준 달성', '다음 라운드로 진행 · 마지막 라운드에서는 다음 문제')
        self._detail('한도 도달', '기준 미달 시 남은 라운드 제외 후 다음 문제')
        self._detail('전체 상한', f'최대 {maximum:,}회 제출 · 조기 종료에 따라 감소')
        self._detail('상태 유지', '같은 문제의 대화·추론 상태 유지 · 다음 문제에서 초기화')
        self._line()
        self._line('모델 입력 · 채점 · 로그', 'bold')
        if fourway:
            self._detail('피드백', '정답 / 오답 / 규칙 위반 / 정답이지만 규칙 위반')
            self._detail('모델 전달', '영어 고정 문구 4종 · 실제 문구는 제출 로그에 표시')
            self._detail('정답 판정', '정답만 인정 · 정답이지만 규칙 위반은 오답으로 처리')
        else:
            self._detail('피드백', 'Correct / Incorrect · 모델 입력은 영어')
        self._detail('출력 방식', 'JSON 요청 · API 스키마 강제 없음 · 로컬 공통 채점')
        self._detail('출력 한도', f"{self.config.get('max_output_tokens', 25000):,} 토큰 · 자동 재시도 0회")
        self._detail('공백 반복', '256자 연속 시 미채점 · 해당 문제 중단 후 다음 문제')
        self._detail('PE 판정', '현재 추가 이동 제한 위반 · 목적지 불일치는 별도 집계')
        self._detail('연구자 화면', '구체적 목표·제한·위반 위치·라운드 정보는 API 입력에 포함하지 않음', 'yellow')
        self._detail('로그', self.directory)
        self._close()

    def round_start(self, episode, round_number):
        self.episode, self.round_number, self.streak = (episode, round_number, 0)
        if self.problems:
            problem = self.problems[episode - 1]
            self.rules = tuple(_field(problem, 'rounds') or _field(problem, 'rules'))
        self.max_attempts, self.required_successes = self._policy(round_number)
        current = self.rules[round_number - 1]
        previous = self.rules[round_number - 2] if round_number > 1 else None
        identity = f"문제 {episode}/{self.config.get('episodes', 1)} · 최종 ID {self._problem_id(episode)}" if self.problems else f"독립 실험 {episode}/{self.config.get('episodes', 1)}"
        self._heading(f'{identity} · 라운드 {round_number}/{len(self.rules)}')
        if self.problems and _field(self.problems[episode - 1], 'condition'):
            self._detail('문제 조건', condition_text(_field(self.problems[episode - 1], 'condition')))
        self._detail('초기 배치', initial_state_text(current, pegs=self.pegs))
        self._detail('배치 읽기', '[아래 → 위] · 매 제출마다 이 배치에서 다시 시작')
        goal = f"{_field(current, 'goal_peg') or 'C'}=[3,2,1]"
        if previous is not None:
            goal += f" · 직전 목표 {_field(previous, 'goal_peg') or 'C'} → 현재 {_field(current, 'goal_peg') or 'C'}"
        self._detail('목표', goal, 'bold')
        old, new = (_restrictions(previous), _restrictions(current))
        self._detail('현재 제한', ' · '.join(new) or '추가 제한 없음 · 기본 하노이 규칙', 'bold')
        if round_number > 1:
            self._detail('직전 제한', ' · '.join(old) or '추가 제한 없음 · 기본 하노이 규칙')
            self._detail('규칙 전환', '직전 제한을 현재 제한으로 교체 · 누적 없음' if old != new else '직전 제한과 동일')
            if initial_state_text(previous, pegs=self.pegs) != initial_state_text(current, pegs=self.pegs):
                self._detail('직전 배치', initial_state_text(previous, pegs=self.pegs))
        self._detail('공통 규칙', '맨 위 원판 1개씩 이동 · 큰 원판을 작은 원판 위에 놓을 수 없음')
        self._line()
        self._detail('제출 한도', f'T = {self.max_attempts}회')
        next_stage = '다음 문제' if round_number == len(self.rules) else '다음 라운드'
        self._detail('성공 기준', f'k = {self.required_successes}회 연속 정답 → {next_stage}')
        if self.max_attempts is not None:
            self._detail('한도 도달', '기준 미달 시 남은 라운드 제외 후 다음 문제')
        self._close()

    def waiting(self, attempt, transport_attempt):
        retry = f' · 통신 시도 {transport_attempt}' if transport_attempt > 1 else ''
        position = f'{attempt:02d}/{self.max_attempts}'
        progress = f"문제 {self.episode}/{self.config.get('episodes', 1)} · " if self.episode is not None else ''
        self._print(self._color(f'  … {progress}R{self.round_number} 시도 {position} · API 응답 대기{retry}', 'dim'))

    def submission(self, record, recovered=False):
        success = bool(record.get('current_success'))
        supplied_streak = record.get('success_streak')
        self.streak = supplied_streak if isinstance(supplied_streak, int) else self.streak + 1 if success else 0
        attempt = record.get('attempt', '?')
        stamp = f'{attempt:02d}' if isinstance(attempt, int) else clean(attempt)
        icon, color = ('✓', 'green') if success else ('✗', 'red')
        feedback = record.get('feedback', 'Correct' if success else 'Incorrect')
        fourway = record.get('feedback_policy', self.config.get('feedback_policy', 'binary_v1')) == 'goal_rule_4way_v1'
        displayed_feedback = {'Correct': '정답', 'Incorrect': '오답', 'Rule violation': '규칙 위반',
                              'Correct, but rule violation': '정답이지만 규칙 위반'}.get(feedback, feedback) if fourway else feedback
        cache = ' · 저장된 기록 복구' if recovered else ''
        position = f'{stamp}/{self.max_attempts}'
        self._print(self._color(f'  {icon} 시도 {position}  {displayed_feedback}  |  연속 정답 {self.streak}/{self.required_successes}{cache}', color))
        if fourway:
            self._print(f'    모델 피드백: {clean(feedback)}')
        audit = record.get('reasoning_replay')
        if audit:
            incoming = audit.get('input', {}).get('opaque_reasoning_items', 0)
            outgoing = audit.get('output', {}).get('opaque_reasoning_items', 0)
            self._print(f'    추론 상태: 이전 {incoming}개 전달 · 이번 {outgoing}개 보존 (암호화/서명 항목)')
        pe = record.get('persistent_error')
        pe_text = '해당 없음 (R1 제외)' if pe is None else '예 · 현재 추가 이동 제한 위반' if pe else '아니오'
        elapsed = record.get('elapsed_seconds')
        duration = f'{elapsed:.2f}초' if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) else '—'
        self._print(f'    지속적 오류 PE: {pe_text}  ·  응답 시간: {duration}')
        if 'movement_rule_error' in record:
            violations = record.get('movement_rule_violations') or []
            invalid_plan = (record.get('base_error') or {}).get('kind') == 'invalid_plan'
            unknown_goal = '판정 불가 (이동 계획 형식 오류)' if invalid_plan else '판정 불가 (기본 이동 오류)'
            goal = '불일치' if record.get('goal_error') is True else '도달' if record.get('goal_error') is False else unknown_goal
            self._print(f'    이동 제한 위반: {len(violations)}건  ·  목적지: {goal}')
            for item in violations:
                move = '→'.join(clean(item.get('move', '')))
                disk = item.get('disk') if item.get('disk') is not None else '확인 불가'
                self._print(f"      {clean(item.get('step'))}번째 {move} · 원판 {clean(disk)}")
        moves = record.get('moves')
        current = self.rules[self.round_number - 1] if isinstance(self.round_number, int) and 1 <= self.round_number <= len(self.rules) else None
        commands = describe_moves(moves, pegs=self.pegs, initial_positions=_field(current, 'initial_positions'))
        if commands is None:
            try:
                empty_plan = json.loads(record.get('raw_output', '')) == {'moves': []}
            except (ValueError, TypeError):
                empty_plan = False
            if empty_plan:
                self._print('    이동 계획: 빈 목록 제출 (0회 이동 · 최소 1회 필요)')
            else:
                self._print('    이동 계획: 유효한 JSON 명령 목록 없음 (원본 응답은 로그에 저장)')
        else:
            self._print(f'    이동 계획 {len(moves)}개 (원판 번호 · 순서):')
            line = ''
            for command in commands:
                candidate = f'{line} · {command}' if line else command
                if line and display_width(candidate) > self.width - 6:
                    self._print('      ' + line)
                    line = ''
                pieces = wrap(command, self.width - 6)
                if len(pieces) > 1:
                    if line:
                        self._print('      ' + line)
                    for piece in pieces[:-1]:
                        self._print('      ' + piece)
                    line = pieces[-1]
                else:
                    line = f'{line} · {command}' if line else command
            if line:
                self._print('      ' + line)
        self._print()

    def round_end(self, round_summary):
        attempts = round_summary.get('attempts', 0)
        criterion_at = round_summary.get('criterion_met_at')
        success = round_summary.get('status') == 'success' or criterion_at is not None
        label = ('첫 정답 달성' if self.required_successes == 1 else f'연속 {self.required_successes}회 정답 달성') if success else f'최대 {self.max_attempts}회 도달 · 기준 미달'
        if not round_summary.get('complete', True):
            label = '생성 중단 · 미완료' if round_summary.get('status') == 'generation_failed' else '진행 중'
        self._print(self._color(f"  R{round_summary.get('round', self.round_number)} 종료  {label}  |  제출 {attempts}회", 'green' if success else 'yellow'))
        if round_summary.get('round', self.round_number) > 1:
            self._print(f"  라운드 PER  {_rate(round_summary.get('persistent_errors', 0), attempts)} · 실제 제출 수를 분모로 사용")
            if 'goal_errors' in round_summary:
                self._print(f"  목적지 불일치 {round_summary.get('goal_errors', 0)}회 · 목적지 판정 불가 {attempts - round_summary.get('goal_evaluated_attempts', 0)}회 (PE와 별도)")
        self._print(f"  라운드 오답률  {_rate(round_summary.get('failures', 0), attempts)}")
        self._print('  ' + '─' * (self.width - 4))

    def generation_failure(self, failure):
        self._line(f"문제 {failure.get('episode', self.episode)}/{self.config.get('episodes', 1)} · 최종 ID {failure['problem_id']} · R{failure['round']} 제출 {failure['attempt']}: 생성 중단 ({failure['reason']})", 'yellow')
        self._line('미채점 · 자동 재시도 없음 · 해당 문제의 남은 라운드 제외 후 다음 문제 진행')

    def problem_exhausted(self, episode, round_number):
        self._print(self._color(f"  문제 {episode}/{self.config.get('episodes', 1)} 종료 · 최종 ID {self._problem_id(episode)} · R{round_number} 한도 내 정답 기준 미달", 'yellow'))
        self._print(self._color('  남은 라운드 미실행 · 다음 문제로 진행', 'yellow'))

    def finish(self, summary):
        error_rounds = 'R1~R5'
        error_problem_count_field = 'error_metric_eligible_problems'
        error_round_count_field = 'eligible_error_rounds'
        self._heading('실험 결과 · 라운드 PER → 문제 MPE → Total MPE')
        completed = summary.get('complete', False)
        self._line(f"상태  {('전체 완료' if completed else '실행 종료 · 미완료 문제 있음' if summary.get('execution_finished') else '진행 중 / 부분 결과')}  |  실제 제출 {summary.get('recorded_submissions', 0)}회", 'bold')
        if summary.get('generation_failure_count'):
            self._line(f"생성 실패 {summary['generation_failure_count']}문제 · 미채점 · 해당 문제의 남은 라운드 제외", 'yellow')
        if summary.get('usage_incomplete'):
            self._line('토큰 사용량은 일부만 관측됨 · 실제 청구액보다 적을 수 있음', 'yellow')
        self._line(f"성공 라운드  {summary.get('successful_rounds', 0)}  |  한도 종료 {summary.get('exhausted_rounds', 0)}  |  미실행 {summary.get('excluded_rounds', 0)}")
        self._line()
        widths = (7, 4, 4, 5, 9, 10, 8, 17)
        headers = ('최종 ID', 'R', '제출', 'O/X', '첫 정답', '기준 달성', '상태', 'PER (PE / 제출)')
        self._line(' '.join((pad(value, width) for value, width in zip(headers, widths))))
        self._line('─' * (sum(widths) + len(widths) - 1))
        for episode in summary.get('episodes', []):
            if episode.get('condition'):
                self._line(f"문제 {episode.get('episode', '?')}/{self.config.get('episodes', 1)} · 최종 ID {episode.get('problem_id')} · {condition_text(episode['condition'])}")
            for item in episode.get('rounds', []):
                first, criterion = (item.get('first_success_attempt'), item.get('criterion_met_at'))
                status = {'success': '성공', 'exhausted': '한도종료', 'in_progress': '진행중', 'pending': '대기', 'generation_failed': '생성중단', 'excluded': '제외'}.get(item.get('status'), '—')
                values = (episode.get('problem_id', f"E{episode.get('episode', '?')}"), f"R{item.get('round', '?')}", str(item.get('attempts', 0)), f"{item.get('successes', 0)}/{item.get('failures', 0)}", f'{first}회' if first is not None else '—', f'{criterion}회' if criterion is not None else '—', status, '제외' if item.get('round') == 1 else _rate(item.get('persistent_errors', 0), item.get('attempts', 0)))
                self._line(' '.join((pad(value, width) for value, width in zip(values, widths))))
            value = episode.get('mpe')
            rate = f'{value:.2%}' if isinstance(value, (float, int)) else '—'
            self._line(f"  {episode.get('problem_id')} MPE {rate} · 포함 라운드 {episode.get('eligible_scored_rounds', 0)}개 (R1 제외)")
            value = episode.get('mean_error_rate')
            rate = f'{value:.2%}' if isinstance(value, (float, int)) and (not isinstance(value, bool)) else '—'
            self._line(f"  {episode.get('problem_id')} 평균 오답률 {rate} · {error_rounds} 진행 라운드 {episode.get(error_round_count_field, '—')}개")
        self._line()
        pooled = summary.get('pooled', {})
        total_mpe = summary.get('total_mpe')
        rate = f'{total_mpe:.2%}' if isinstance(total_mpe, (float, int)) else '—'
        self._line(f"Total MPE  {rate} · R2 이상 진행된 문제 N={summary.get('metric_eligible_problems', 0)}", 'bold')
        value = summary.get('total_mean_error_rate')
        rate = f'{value:.2%}' if isinstance(value, (float, int)) and (not isinstance(value, bool)) else '—'
        self._line(f"전체 평균 오답률  {rate} · {error_rounds} 실제 제출이 있는 문제 N={summary.get(error_problem_count_field, '—')}", 'bold')
        self._line(f"참고 PER (pooled)  {_rate(pooled.get('persistent_errors', 0), pooled.get('attempts', 0))}")
        self._line(f"참고 오답률 (pooled)  {_rate(pooled.get('failures', 0), pooled.get('attempts', 0))} · R2~R5 오답 합 / 제출 합")
        if 'goal_errors' in pooled:
            self._line(f"목적지 불일치 {pooled.get('goal_errors', 0)}회 · 목적지 판정 불가 {pooled.get('attempts', 0) - pooled.get('goal_evaluated_attempts', 0)}회 (PE와 별도)")
        for condition, item in summary.get('conditions', {}).items():
            value = item.get('mpe')
            rate = f'{value:.2%}' if isinstance(value, (float, int)) else '—'
            self._line(f"{item.get('label', condition_text(condition))}  MPE {rate} · 포함 문제 N={item.get('metric_eligible_problems', 0)}")
            value = item.get('mean_error_rate')
            rate = f'{value:.2%}' if isinstance(value, (float, int)) and (not isinstance(value, bool)) else '—'
            self._line(f"  평균 오답률 {rate} · {error_rounds} 문제 N={item.get(error_problem_count_field, '—')}")
        self._line('PE = 현재 추가 이동 제한을 하나 이상 위반한 제출 (제출당 최대 1회)')
        self._line('목적지 불일치는 PE에서 제외 · 직전 라운드 성공 여부와 무관')
        self._line('성공한 제출도 분모에 포함 · 미사용 시도와 R1은 지속적 오류율에서 제외')
        self._line('문제 MPE = 진행된 R2~R5 PER의 산술평균 · Total MPE = 포함 문제 MPE의 산술평균')
        self._line(f'오답률: {error_rounds} 라운드 → 문제 → 전체/유형 순서로 산술평균 · 미실행 라운드 제외')
        self._line(f'상세 로그  {self.directory}')
        self._close()
