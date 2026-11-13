"""Keep scoring and aggregation fixed across feedback and difficulty conditions.

Update this contract only after explicit user approval for a metric change.
"""

import ast
import copy
from hashlib import sha256
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MetricContractTests(unittest.TestCase):
    def test_scoring_and_aggregation_match_the_approved_contract(self):
        expected = json.loads((ROOT / 'tests/fixtures/metric_contract.json').read_text())
        module = ast.parse((ROOT / 'src/evaluate.py').read_text())
        found = {}
        for source in module.body:
            if not isinstance(source, (ast.FunctionDef, ast.ClassDef)) or source.name not in expected:
                continue
            node = copy.deepcopy(source)
            if node.name == 'grade_response':
                # Only message construction is excluded from the scoring fingerprint.
                kept = []
                for statement in node.body:
                    assigned = {n.id for n in ast.walk(statement)
                                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                    if isinstance(statement, ast.If) and 'feedback' in assigned:
                        self.assertLessEqual(assigned, {'feedback', 'unit', 'positions'},
                                             'The feedback block must not change scoring state.')
                    else:
                        kept.append(statement)
                node.body = kept
            found[node.name] = sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        found['constants'] = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in expected['constants']
        }
        # JSON normalizes tuples to lists, matching the stored contract.
        self.assertEqual(json.loads(json.dumps(found)), expected,
                         'Scoring or aggregation changed: obtain explicit user approval first.')


if __name__ == '__main__':
    unittest.main()
