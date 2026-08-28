import unittest
from unittest.mock import MagicMock

from single_turn_agent import ASTContextAnalyzer, SingleTurnCodeAgent


class TestASTContextAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ASTContextAnalyzer()

    def test_empty_snippet_returns_empty_sets(self):
        result = self.analyzer.extract_context("   \n  ")
        self.assertEqual(result, {"variables": set(), "imports": set()})

    def test_simple_assignment_is_captured(self):
        result = self.analyzer.extract_context("x = 5")
        self.assertEqual(result["variables"], {"x"})

    def test_multiple_assignments_are_all_captured(self):
        code = "a = 1\nb = 2\nc = a + b"
        result = self.analyzer.extract_context(code)
        self.assertEqual(result["variables"], {"a", "b", "c"})

    def test_import_statement_is_captured_verbatim(self):
        result = self.analyzer.extract_context("import os")
        self.assertEqual(result["imports"], {"import os"})

    def test_from_import_statement_is_captured_verbatim(self):
        result = self.analyzer.extract_context("from typing import List, Dict")
        self.assertEqual(result["imports"], {"from typing import List, Dict"})

    def test_combined_imports_and_variables(self):
        code = "import os\nfrom typing import List\nvalue = os.getcwd()"
        result = self.analyzer.extract_context(code)
        self.assertEqual(result["variables"], {"value"})
        self.assertEqual(result["imports"], {"import os", "from typing import List"})

    def test_attribute_assignment_is_not_captured(self):
        """`self.x = 5` a pour left_node un noeud "attribute", pas "identifier" —
        l'implémentation actuelle ne le détecte donc pas comme variable."""
        result = self.analyzer.extract_context("self.x = 5")
        self.assertEqual(result["variables"], set())

    def test_tuple_unpacking_is_not_captured(self):
        """`a, b = 1, 2` a pour left_node un "pattern_list", pas "identifier" —
        non détecté non plus par l'implémentation actuelle."""
        result = self.analyzer.extract_context("a, b = 1, 2")
        self.assertEqual(result["variables"], set())

    def test_nested_function_assignments_are_all_captured_without_scoping(self):
        """L'extraction parcourt tout l'arbre sans s'arrêter aux frontières de
        fonction/classe : les variables de la fonction imbriquée remontent
        telles quelles, sans distinction de portée."""
        code = (
            "def outer():\n"
            "    outer_var = 1\n"
            "    def inner():\n"
            "        inner_var = 2\n"
            "        return inner_var\n"
            "    return outer_var\n"
        )
        result = self.analyzer.extract_context(code)
        self.assertEqual(result["variables"], {"outer_var", "inner_var"})


class TestBuildStructuredPrompt(unittest.TestCase):
    def setUp(self):
        self.agent = SingleTurnCodeAgent(llm_client=MagicMock())

    def test_prompt_contains_file_path(self):
        prompt = self.agent.build_structured_prompt("app/main.py", "x = 1", [])
        self.assertIn("FILE PATH: app/main.py", prompt)

    def test_prompt_lists_extracted_imports_and_variables(self):
        code = "import os\nresult = os.getcwd()"
        prompt = self.agent.build_structured_prompt("app/main.py", code, [])
        self.assertIn("import os", prompt)
        self.assertIn("result", prompt)

    def test_prompt_uses_none_placeholder_when_no_imports_or_variables(self):
        prompt = self.agent.build_structured_prompt("app/main.py", "print('hi')", [])
        self.assertIn("IMPORTS:\nNone", prompt)
        self.assertIn("ACTIVE SCOPE VARIABLES:\nNone", prompt)

    def test_prompt_includes_retrieved_chunks(self):
        chunks = [
            {"file_path": "utils.py", "raw_code": "def helper():\n    pass"},
            {"file_path": "models.py", "raw_code": "class Model:\n    pass"},
        ]
        prompt = self.agent.build_structured_prompt("app/main.py", "x = 1", chunks)
        self.assertIn("# Example 1 | File: utils.py", prompt)
        self.assertIn("def helper():", prompt)
        self.assertIn("# Example 2 | File: models.py", prompt)
        self.assertIn("class Model:", prompt)

    def test_prompt_handles_chunks_missing_expected_keys(self):
        chunks = [{}]
        prompt = self.agent.build_structured_prompt("app/main.py", "x = 1", chunks)
        self.assertIn("# Example 1 | File: unknown", prompt)

    def test_prompt_ends_with_unfinished_code(self):
        prompt = self.agent.build_structured_prompt("app/main.py", "def foo(", [])
        self.assertTrue(prompt.rstrip("\n").endswith("def foo("))


class TestGenerateCompletion(unittest.TestCase):
    def _make_mock_client(self, completion_text: str) -> MagicMock:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=completion_text))
        ]
        return mock_client

    def test_returns_llm_response_content(self):
        mock_client = self._make_mock_client("    return x + 1")
        agent = SingleTurnCodeAgent(llm_client=mock_client)

        result = agent.generate_completion("app/main.py", "def add(x):", [])

        self.assertEqual(result, "    return x + 1")

    def test_calls_llm_with_expected_model_and_generation_params(self):
        mock_client = self._make_mock_client("...")
        agent = SingleTurnCodeAgent(llm_client=mock_client, model_name="custom-model")

        agent.generate_completion("app/main.py", "x = 1", [])

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "custom-model")
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["max_tokens"], 150)

    def test_sends_system_and_user_messages(self):
        mock_client = self._make_mock_client("...")
        agent = SingleTurnCodeAgent(llm_client=mock_client)

        agent.generate_completion("app/main.py", "x = 1", [])

        _, kwargs = mock_client.chat.completions.create.call_args
        messages = kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Règles strictes", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("FILE PATH: app/main.py", messages[1]["content"])

    def test_user_message_matches_build_structured_prompt(self):
        mock_client = self._make_mock_client("...")
        agent = SingleTurnCodeAgent(llm_client=mock_client)
        chunks = [{"file_path": "utils.py", "raw_code": "def helper(): pass"}]

        expected_prompt = agent.build_structured_prompt("app/main.py", "x = 1", chunks)
        agent.generate_completion("app/main.py", "x = 1", chunks)

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["messages"][1]["content"], expected_prompt)

    def test_default_model_name_is_used_when_not_overridden(self):
        mock_client = self._make_mock_client("...")
        agent = SingleTurnCodeAgent(llm_client=mock_client)

        agent.generate_completion("app/main.py", "x = 1", [])

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "qwen2.5-coder-7b")


if __name__ == "__main__":
    unittest.main()
