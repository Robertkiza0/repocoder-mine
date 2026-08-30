import unittest

from container_ast_chunker import chunk_code_by_ast


class TestChunkCodeByAst(unittest.TestCase):
    def test_small_function_becomes_one_whole_chunk(self):
        code = "def add(a, b):\n    return a + b\n"
        chunks = chunk_code_by_ast(code, max_lines=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["type"], "function_definition")
        self.assertIn("def add(a, b):", chunks[0]["code"])
        self.assertIn("return a + b", chunks[0]["code"])

    def test_large_flat_function_is_not_silently_dropped(self):
        """Le bug de l'exemple original : une fonction trop grande sans
        fonction/classe imbriquée ne produisait AUCUN chunk. Ici son contenu
        doit être intégralement couvert par les sous-chunks."""
        body_lines = "\n".join(f"    x{i} = {i}" for i in range(40))
        code = f"def big_function():\n{body_lines}\n    return x39\n"
        chunks = chunk_code_by_ast(code, max_lines=15, overlap_lines=2)

        self.assertGreater(len(chunks), 1)  # doit être découpé, pas ignoré
        for chunk in chunks:
            self.assertEqual(chunk["type"], "function_definition:partial")
            self.assertIn("def big_function():", chunk["code"])  # en-tête conservé partout

        # tout le contenu (chaque x{i} = {i}) doit apparaître dans au moins un sous-chunk
        for i in range(40):
            self.assertTrue(
                any(f"x{i} = {i}" in c["code"] for c in chunks),
                f"ligne x{i} = {i} perdue",
            )

    def test_split_never_cuts_a_statement_in_half(self):
        body_lines = []
        for i in range(10):
            body_lines.append(f"    if x == {i}:")
            body_lines.append(f"        y = {i}")
            body_lines.append(f"        z = {i} * 2")
        code = "def f(x):\n" + "\n".join(body_lines) + "\n    return y\n"
        chunks = chunk_code_by_ast(code, max_lines=12, overlap_lines=2)

        # chaque instruction "if x == N:" doit apparaître avec son corps complet
        # (y = N ET z = N * 2) dans le MÊME sous-chunk, jamais coupée entre deux
        for i in range(10):
            containing = [c for c in chunks if f"if x == {i}:" in c["code"]]
            self.assertTrue(containing, f"instruction if x == {i} introuvable")
            for c in containing:
                self.assertIn(f"y = {i}", c["code"])
                self.assertIn(f"z = {i} * 2", c["code"])

    def test_overlap_repeats_tail_of_previous_group(self):
        body_lines = "\n".join(f"    x{i} = {i}" for i in range(20))
        code = f"def f():\n{body_lines}\n"
        chunks = chunk_code_by_ast(code, max_lines=10, overlap_lines=3)

        self.assertGreater(len(chunks), 1)
        # la fin du 1er sous-chunk doit réapparaître au début du corps du 2e
        # (chevauchement pour garder les variables locales actives visibles)
        first_lines = chunks[0]["code"].splitlines()
        second_code = chunks[1]["code"]
        tail_of_first = first_lines[-1].strip()
        self.assertIn(tail_of_first, second_code)

    def test_decorated_function_keeps_decorators_in_header(self):
        body_lines = "\n".join(f"    x{i} = {i}" for i in range(30))
        code = f"@staticmethod\n@another_decorator\ndef f():\n{body_lines}\n"
        chunks = chunk_code_by_ast(code, max_lines=10, overlap_lines=2)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("@staticmethod", chunk["code"])
            self.assertIn("@another_decorator", chunk["code"])
            self.assertIn("def f():", chunk["code"])

    def test_small_decorated_function_is_one_chunk(self):
        code = "@property\ndef value(self):\n    return self._value\n"
        chunks = chunk_code_by_ast(code, max_lines=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["type"], "decorated_definition")
        self.assertIn("@property", chunks[0]["code"])

    def test_class_with_small_methods_is_one_chunk(self):
        code = (
            "class Foo:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "\n"
            "    def bar(self):\n"
            "        return self.x\n"
        )
        chunks = chunk_code_by_ast(code, max_lines=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["type"], "class_definition")
        self.assertIn("def __init__", chunks[0]["code"])
        self.assertIn("def bar", chunks[0]["code"])

    def test_large_class_splits_by_method_keeping_class_header(self):
        methods = []
        for i in range(6):
            methods.append(f"    def method_{i}(self):\n        return {i}\n")
        code = "class Big:\n" + "\n".join(methods)
        chunks = chunk_code_by_ast(code, max_lines=8, overlap_lines=1)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk["type"], "class_definition:partial")
            self.assertIn("class Big:", chunk["code"])  # en-tête (signature de classe) conservé

        for i in range(6):
            self.assertTrue(
                any(f"def method_{i}" in c["code"] for c in chunks),
                f"method_{i} perdue",
            )

    def test_module_level_orphan_code_is_not_dropped(self):
        code = (
            "import os\n"
            "import sys\n"
            "\n"
            "CONSTANT = 42\n"
            "\n"
            "def f():\n"
            "    return CONSTANT\n"
        )
        chunks = chunk_code_by_ast(code, max_lines=30)
        module_chunks = [c for c in chunks if c["type"] == "module"]
        self.assertTrue(module_chunks, "code de niveau module (imports/constante) perdu")
        combined = "\n".join(c["code"] for c in module_chunks)
        self.assertIn("import os", combined)
        self.assertIn("import sys", combined)
        self.assertIn("CONSTANT = 42", combined)

        function_chunks = [c for c in chunks if c["type"] == "function_definition"]
        self.assertEqual(len(function_chunks), 1)

    def test_empty_code_returns_no_chunks(self):
        self.assertEqual(chunk_code_by_ast("   \n  "), [])

    def test_chunks_sorted_by_start_line(self):
        code = (
            "import os\n"
            "\n"
            "def f():\n"
            "    return 1\n"
            "\n"
            "CONSTANT = 1\n"
        )
        chunks = chunk_code_by_ast(code, max_lines=30)
        starts = [c["start_line"] for c in chunks]
        self.assertEqual(starts, sorted(starts))


if __name__ == "__main__":
    unittest.main()
