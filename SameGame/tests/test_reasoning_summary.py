"""Opt-in diagnostic summaries must not change grading or public prompts."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import diagnostics, model_runner, providers
from tests.test_model_runner import problem, response


def summarized(text="Diagnostic summary.", *, marker="opaque-state"):
    raw = response(marker=marker)
    if text is not None:
        raw["output"][0]["summary"] = [{"type": "summary_text", "text": text}]
    return raw


class ReasoningSummaryTests(unittest.TestCase):
    def setUp(self):
        self.settings = model_runner.load_settings(profile="sol")
        self.settings.update(max_attempts=2, required_streak=1, reasoning_summary="auto")

    def execute(self, items, directory, receive, *, resume=False, settings=None):
        with redirect_stderr(StringIO()):
            return model_runner.execute(items, settings or self.settings, directory,
                                        resume=resume, client=object(), receive=receive)

    def cli(self, arguments):
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            result = model_runner.main(arguments)
        return result, output.getvalue(), errors.getvalue()

    def test_option_only_adds_summary_parameter_and_preserves_default_request(self):
        settings = model_runner.load_settings(profile="sol")
        history = [{"role": "user", "content": "Public input only."}]
        original = deepcopy(history)
        baseline = providers.make_request(settings, "Public instructions.", history)
        self.assertEqual(diagnostics.mode(settings), "off")
        self.assertEqual(diagnostics.make_request(settings, "Public instructions.", history), baseline)
        for mode in ("auto", "concise", "detailed"):
            enabled = diagnostics.make_request(dict(settings, reasoning_summary=mode),
                                               "Public instructions.", history)
            self.assertEqual(enabled["reasoning"].pop("summary"), mode)
            self.assertEqual(enabled, baseline)
        self.assertEqual(history, original)
        for profile in ("opus", "sonnet"):
            other = model_runner.load_settings(profile=profile)
            self.assertEqual(diagnostics.make_request(other, "system", history),
                             providers.make_request(other, "system", history))
            with self.assertRaises(ValueError):
                diagnostics.make_request(dict(other, reasoning_summary="auto"), "system", history)

    def test_extracts_only_api_summary_text_from_every_reasoning_item(self):
        raw = summarized("First summary.")
        raw["output"].insert(1, {
            "type": "reasoning", "encrypted_content": "another-opaque-state",
            "summary": [{"type": "summary_text", "text": "Second summary."}],
        })
        raw["output"][0]["summary"].append({"type": "unrecognized", "text": "Not a summary."})
        result = diagnostics.extract_summary(raw)
        self.assertIn("First summary.", result)
        self.assertIn("Second summary.", result)
        self.assertLess(result.index("First summary."), result.index("Second summary."))
        for excluded in ("opaque-state", "another-opaque-state", "Not a summary.", '"moves"'):
            self.assertNotIn(excluded, result)
        self.assertEqual(diagnostics.extract_summary(summarized(None)), "")

    def test_summary_logs_capture_pre_advance_round_and_preserve_native_history(self):
        requests, originals = [], []
        first = problem("one", 2)
        first["rounds"][1]["forbidden"] = "BB"

        def receive(client, provider, request, path):
            requests.append(deepcopy(request))
            raw = summarized(f"Summary for call {len(requests)}.", marker=f"opaque-{len(requests)}")
            originals.append(deepcopy(raw))
            return raw

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = self.execute([first, problem("two")], directory, receive)
            self.assertTrue(report["complete"])
            self.assertEqual([row["feedback"] for row in report["submissions"]], ["Correct"] * 3)
            saved = [model_runner.read_json(path) for path in sorted(
                (directory / "submissions").glob("*_reasoning_summary.json"))]
            self.assertEqual([(row["id"], row["round"], row["attempt"]) for row in saved],
                             [("one", 1, 1), ("one", 2, 1), ("two", 1, 1)])
            self.assertEqual([row["forbidden"] for row in saved], ["AA", "BB", "AA"])
            self.assertTrue(all(row["summary_mode"] == "auto" for row in saved))
            self.assertTrue(all(row["summary_status"] == "available" for row in saved))
            self.assertTrue(all(row["input"] == "AABB" and row["target"] == 0 for row in saved))
            self.assertTrue(all(row["feedback"] == "Correct" for row in saved))
            log = (directory / "reasoning_log.txt").read_text()
            for number in (1, 2, 3):
                self.assertEqual(log.count(f"Summary for call {number}."), 1)
            self.assertNotIn("opaque-", log)
            self.assertEqual(requests[1]["input"][1:3], providers.continuation_items(originals[0], "openai"))
            self.assertEqual(len(requests[2]["input"]), 1)
            self.assertEqual(requests[1]["input"][-2], {"role": "user", "content": "Correct"})
            self.assertTrue(all(request["reasoning"]["summary"] == "auto" for request in requests))
            self.assertEqual(model_runner.read_json(directory / "protocol.json")["settings"]["reasoning_summary"], "auto")

    def test_empty_summary_is_recorded_without_changing_score_or_retrying(self):
        calls = []

        def receive(*args):
            calls.append(1)
            return summarized(None)

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = self.execute([problem("one")], directory, receive)
            self.assertEqual(calls, [1])
            self.assertTrue(report["complete"])
            self.assertEqual(report["submissions"][0]["feedback"], "Correct")
            saved = model_runner.read_json(directory / "submissions" / "00001_reasoning_summary.json")
            self.assertEqual(saved["summary"], "")
            self.assertEqual(saved["summary_status"], "empty")
            self.assertTrue((directory / "reasoning_log.txt").read_text().strip())
            self.assertNotIn("opaque-state", (directory / "reasoning_log.txt").read_text())

    def test_disabled_mode_does_not_create_diagnostic_files(self):
        settings = dict(self.settings, reasoning_summary="off")
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = self.execute([problem("one")], directory, lambda *args: summarized(), settings=settings)
            self.assertTrue(report["complete"])
            self.assertFalse((directory / "reasoning_log.txt").exists())
            self.assertFalse(list((directory / "submissions").glob("*_reasoning_summary.json")))

    def test_resume_reuses_response_and_rebuilds_log_without_duplicate_summary(self):
        calls = []

        def receive(*args):
            calls.append(1)
            return summarized("Unique recovery diagnostic.")

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            real_save = model_runner.save_progress

            def interrupt_before_checkpoint(directory, session, item, history):
                if session.responses:
                    raise OSError("simulated checkpoint interruption")
                return real_save(directory, session, item, history)

            with patch.object(model_runner, "save_progress", side_effect=interrupt_before_checkpoint):
                with self.assertRaises(OSError):
                    self.execute([problem("one")], directory, receive)
            self.assertEqual(model_runner.read_json(directory / "checkpoint.json")["responses"], [])
            report = self.execute([problem("one")], directory, receive, resume=True)
            self.assertTrue(report["complete"])
            self.assertEqual(len(calls), 1)
            self.assertEqual((directory / "reasoning_log.txt").read_text().count("Unique recovery diagnostic."), 1)
            self.assertEqual(len(list((directory / "submissions").glob("*_reasoning_summary.json"))), 1)
            (directory / "reasoning_log.txt").unlink()
            self.execute([problem("one")], directory,
                         lambda *args: self.fail("completed run regenerated"), resume=True)
            self.assertEqual((directory / "reasoning_log.txt").read_text().count("Unique recovery diagnostic."), 1)

    def test_cli_supports_bare_flag_and_config_override_and_rejects_other_providers(self):
        result, output, _ = self.cli(["--profile", "sol", "--limit", "1", "--reasoning-summary", "--dry-run"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output)["request"]["reasoning"]["summary"], "auto")
        result, output, _ = self.cli(["--profile", "sol", "--limit", "1", "--dry-run"])
        self.assertEqual(result, 0)
        self.assertNotIn("summary", json.loads(output)["request"]["reasoning"])
        result, _, error = self.cli(["--profile", "opus", "--reasoning-summary", "--dry-run"])
        self.assertEqual(result, 1)
        self.assertTrue(error.strip())
        with TemporaryDirectory() as temporary:
            config = Path(temporary) / "openai.toml"
            contents = model_runner.DEFAULT_CONFIGS["sol"].read_text()
            contents = contents.replace('dataset = "../data/problems.json"',
                                        'dataset = ' + json.dumps(str(model_runner.PROJECT_DIR / "data" / "problems.json")))
            config.write_text(contents + '\nreasoning_summary = "detailed"\n')
            self.assertEqual(model_runner.load_settings(config)["reasoning_summary"], "detailed")
            result, output, _ = self.cli(["--config", str(config), "--limit", "1", "--reasoning-summary", "off", "--dry-run"])
            self.assertEqual(result, 0)
            self.assertNotIn("summary", json.loads(output)["request"]["reasoning"])

    def test_resume_uses_frozen_summary_option_and_rejects_override(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def fail_transport(*args):
                raise providers.ProviderError("interrupted")

            with self.assertRaises(providers.ProviderError):
                self.execute([problem("one")], directory, fail_transport)
            result, output, _ = self.cli(["--resume", str(directory), "--dry-run"])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output)["request"]["reasoning"]["summary"], "auto")
            result, _, error = self.cli(["--resume", str(directory), "--reasoning-summary", "off", "--dry-run"])
            self.assertEqual(result, 1)
            self.assertTrue(error.strip())
        self.assertEqual(model_runner.implementation_fingerprint(),
                         model_runner.implementation_fingerprint(dict(self.settings, reasoning_summary="off")))
        self.assertNotEqual(model_runner.implementation_fingerprint(),
                            model_runner.implementation_fingerprint(self.settings))


if __name__ == "__main__":
    unittest.main()
