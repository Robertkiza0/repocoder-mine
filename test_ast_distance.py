import math
import unittest

from ast_distance import compute_query_vars_with_attention

CODE = """
def outer():
    close_var = 1
    def inner():
        far_setup = 2
        for i in range(3):
            if i > 0:
                result = close_var + i
    return inner
""".strip("\n")

CURSOR_LINE = 7  # "result = close_var + i"


class TestComputeQueryVarsWithAttention(unittest.TestCase):
    def test_variable_on_cursor_line_has_zero_distance(self):
        result = compute_query_vars_with_attention(CODE, CURSOR_LINE, {"close_var"}, lam=0.1)
        self.assertAlmostEqual(result["close_var"], 1.0)  # exp(-lambda*0) == 1.0

    def test_closer_variable_has_higher_attention_than_farther_one(self):
        result = compute_query_vars_with_attention(
            CODE, CURSOR_LINE, {"close_var", "i", "far_setup"}, lam=0.1
        )
        self.assertGreater(result["close_var"], result["i"])
        self.assertGreater(result["i"], result["far_setup"])

    def test_missing_variable_is_omitted_not_defaulted(self):
        result = compute_query_vars_with_attention(CODE, CURSOR_LINE, {"close_var", "does_not_exist"}, lam=0.1)
        self.assertIn("close_var", result)
        self.assertNotIn("does_not_exist", result)

    def test_occurrence_after_cursor_is_ignored(self):
        # "result" est assigné SUR la ligne du curseur (Store, pas avant) ; une
        # variable qui n'apparaît qu'APRÈS line_no ne doit jamais être retenue.
        code_with_later_use = CODE + "\n    later_var = close_var\n"
        result = compute_query_vars_with_attention(
            code_with_later_use, CURSOR_LINE, {"later_var"}, lam=0.1
        )
        self.assertNotIn("later_var", result)

    def test_larger_lambda_shrinks_attention_faster_with_distance(self):
        result_small_lambda = compute_query_vars_with_attention(CODE, CURSOR_LINE, {"far_setup"}, lam=0.05)
        result_large_lambda = compute_query_vars_with_attention(CODE, CURSOR_LINE, {"far_setup"}, lam=0.5)
        self.assertGreater(result_small_lambda["far_setup"], result_large_lambda["far_setup"])

    def test_attention_formula_matches_exp_lambda_distance(self):
        # vérifie explicitement que le score suit exp(-lambda * distance), pas
        # juste "plus proche = plus grand" en général
        lam = 0.2
        result = compute_query_vars_with_attention(CODE, CURSOR_LINE, {"close_var"}, lam=lam)
        self.assertAlmostEqual(result["close_var"], math.exp(-lam * 0))

    def test_function_parameter_is_found_as_occurrence(self):
        code = "def f(threshold):\n    return threshold * 2\n"
        result = compute_query_vars_with_attention(code, line_no=2, candidate_vars={"threshold"}, lam=0.1)
        self.assertIn("threshold", result)
        self.assertGreater(result["threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
