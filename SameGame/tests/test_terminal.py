"""Terminal diagnostics keep correctness, private PE, and API usage distinct."""

from contextlib import redirect_stderr
from io import StringIO
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.model_runner import Terminal


class TtyBuffer(StringIO):
    def isatty(self):
        return True


class TerminalTests(unittest.TestCase):
    settings = {"label": "GPT-6.1 Sol · medium", "provider": "openai", "reasoning_summary": "off",
                "max_attempts": 10, "required_streak": 2}

    @staticmethod
    def row(persistent=True):
        return {"feedback": "Correct, but rule violation" if persistent else "Correct",
                "correct": not persistent, "moves": "B1 B1 A1" if persistent else "A1 A1 B1", "answer": "",
                "success_streak": 0 if persistent else 1, "violation_steps": [2] if persistent else [],
                "persistent_error": persistent, "previous_rule_compliant": persistent,
                "previous_violation_steps": None if persistent is None else [] if persistent else [2]}

    def test_previous_rule_compliant_error_and_tokens_are_named(self):
        with redirect_stderr(StringIO()) as output:
            terminal = Terminal(self.settings, [object()], Path("runs/example"))
            terminal.submission(self.row(), 24.8, {"input_tokens": 1734, "output_tokens": 1021})
        lines = output.getvalue().splitlines()
        self.assertTrue(any("정답 판정" in line and "오답" in line for line in lines))
        self.assertTrue(any("지속 오류" in line and "발생 · PE=1" in line for line in lines))
        self.assertTrue(any("현재 규칙 위반" in line and "2번째 행동" in line for line in lines))
        self.assertTrue(any("이전 규칙 위반" in line and "없음" in line for line in lines))
        self.assertIn("입력 1,734 / 출력 1,021", output.getvalue())
        self.assertNotIn("1,734 → 1,021", output.getvalue())
        self.assertNotIn("\033", output.getvalue())

    def test_pe_absence_and_first_round_exclusion_have_distinct_labels(self):
        for persistent, expected in ((False, "없음 · PE=0"), (None, "집계 제외 · R1")):
            with self.subTest(persistent=persistent), redirect_stderr(StringIO()) as output:
                terminal = Terminal(self.settings, [object()], Path("runs/example"))
                terminal.submission(self.row(persistent), 1, {"input_tokens": 1, "output_tokens": 2})
            self.assertIn(expected, output.getvalue())
            if persistent is None:
                self.assertNotIn("PE=0", output.getvalue())

    def test_no_color_tty_wraps_hangul_paths_and_long_answers_to_terminal_cells(self):
        record = {"input": "AA BB " * 5 + "CC", "target": 2, "forbidden": "BB"}
        session = SimpleNamespace(item_index=0, round_index=1, attempt=2, current_record=record,
                                  current_item={"id": "SG-026", "rounds": [{"forbidden": "AA"}, record]})
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((64, 24))), \
                patch.dict(os.environ, {"NO_COLOR": "1", "TERM": "xterm-256color"}), \
                redirect_stderr(TtyBuffer()) as output:
            terminal = Terminal(self.settings, [object()], Path("/아주긴저장경로" * 12))
            with patch.object(terminal, "possible_answers", return_value=("A A", "B C", "D E", "EA", "EB")):
                terminal.start_attempt(session)
            terminal.submission(dict(self.row(), moves="A1 B1 C1 D1 E1 " * 5), 1,
                                {"input_tokens": 100, "output_tokens": 200})
            with terminal.waiting(session):
                pass
        self.assertNotIn("\033", output.getvalue())
        self.assertTrue(all(Terminal.display_width(line) <= 64 for line in output.getvalue().splitlines()))
        self.assertIn('"A A"', output.getvalue())
        self.assertEqual(Terminal.display_width("한글e\u0301"), 5)

    def test_summary_excludes_unscored_rounds_from_pe_count(self):
        with redirect_stderr(StringIO()) as output:
            terminal = Terminal(self.settings, [object()], Path("runs/example"))
            terminal.summary({"complete": True, "submissions": [self.row(None), self.row(True), self.row(False)],
                              "summary": {"mpe": .5}})
        self.assertIn("1건 / 집계 제출 2건", output.getvalue())
        self.assertIn("50.00% · 고정 T 기준 · 문제별 평균", output.getvalue())


if __name__ == "__main__":
    unittest.main()
