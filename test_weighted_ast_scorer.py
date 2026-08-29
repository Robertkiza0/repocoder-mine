import math
import time
import unittest

from weighted_ast_scorer import weighted_ast_attention_score


class TestWeightedAstAttentionScore(unittest.TestCase):
    def test_both_empty_returns_zero(self):
        score = weighted_ast_attention_score({}, set(), set(), set(), {})
        self.assertEqual(score, 0.0)

    def test_empty_query_nonempty_chunk_returns_zero(self):
        # union > 0 (poids du chunk), intersection = 0 -> 0.0, pas de division par zéro
        score = weighted_ast_attention_score({}, set(), {"x", "y"}, {"os"}, {})
        self.assertEqual(score, 0.0)

    def test_perfect_match_identical_sets_returns_one(self):
        # aucun symbole "en trop" d'un côté ou de l'autre -> intersection == union
        query_vars = {"model": 0.9, "batch": 0.5}
        query_imports = {"torch"}
        chunk_symbols = {"model", "batch"}
        chunk_imports = {"torch"}
        score = weighted_ast_attention_score(query_vars, query_imports, chunk_symbols, chunk_imports, {})
        self.assertEqual(score, 1.0)

    def test_extra_chunk_symbols_lower_the_score(self):
        """Différence clé avec le rappel pondéré précédent : un chunk "bruité"
        (avec des symboles hors-sujet) est maintenant pénalisé, puisqu'il
        gonfle l'union sans contribuer à l'intersection."""
        query_vars = {"model": 0.9, "batch": 0.5}
        query_imports = {"torch"}
        chunk_imports = {"torch"}

        exact_chunk = {"model", "batch"}
        noisy_chunk = {"model", "batch", "unrelated_1", "unrelated_2"}

        exact_score = weighted_ast_attention_score(query_vars, query_imports, exact_chunk, chunk_imports, {})
        noisy_score = weighted_ast_attention_score(query_vars, query_imports, noisy_chunk, chunk_imports, {})

        self.assertEqual(exact_score, 1.0)
        self.assertLess(noisy_score, exact_score)

    def test_no_overlap_returns_zero(self):
        query_vars = {"model": 0.9}
        query_imports = {"torch"}
        chunk_symbols = {"unrelated"}
        chunk_imports = {"numpy"}
        score = weighted_ast_attention_score(query_vars, query_imports, chunk_symbols, chunk_imports, {})
        self.assertEqual(score, 0.0)

    def test_partial_match_exact_arithmetic(self):
        # doc_weights par défaut à 1.0 (absents du dict)
        query_vars = {"a": 1.0, "b": 1.0}  # poids côté requête: 1.0*1.0*2.0=2.0 chacun
        query_imports = {"x", "y"}  # poids côté requête: 1.0*2.5=2.5 chacun
        chunk_symbols = {"a", "extra_symbol"}  # "a" matche, "extra_symbol" est chunk-seul (poids 1.0)
        chunk_imports = {"x"}  # "x" matche, rien d'autre côté chunk

        # union = a(2.0) + b(2.0) + x(2.5) + y(2.5) [côté requête, tous comptés]
        #       + extra_symbol(1.0) [chunk-seul, IDF par défaut, pas de type_boost]
        #       = 10.0
        # intersection = a(2.0, matché) + x(2.5, matché) = 4.5
        score = weighted_ast_attention_score(query_vars, query_imports, chunk_symbols, chunk_imports, {})
        self.assertAlmostEqual(score, 4.5 / 10.0)

    def test_chunk_only_symbols_use_doc_weight_without_type_boost(self):
        # symbole chunk-seul : poids = doc_weights.get(v, 1.0), PAS multiplié par var_weight/import_weight
        query_vars = {"a": 1.0}
        chunk_symbols = {"a", "b"}
        doc_weights = {"b": 4.0}

        score = weighted_ast_attention_score(query_vars, set(), chunk_symbols, set(), doc_weights)
        # union = a(1.0*1.0*2.0=2.0, côté requête) + b(4.0, chunk-seul, IDF brut)= 6.0
        # intersection = a(2.0, matché)
        self.assertAlmostEqual(score, 2.0 / 6.0)

    def test_doc_weights_scale_query_side_contribution(self):
        query_vars = {"rare_symbol": 1.0}
        chunk_symbols = {"rare_symbol"}
        doc_weights = {"rare_symbol": 3.0}
        score = weighted_ast_attention_score(query_vars, set(), chunk_symbols, set(), doc_weights)
        # seul symbole des deux côtés, matché -> intersection == union quel que soit le poids
        self.assertEqual(score, 1.0)

    def test_attention_score_affects_relative_weight_not_bounds(self):
        query_vars = {"close_var": 0.9, "far_var": 0.1}
        query_imports = set()
        chunk_imports = set()

        score_close_matched = weighted_ast_attention_score(
            query_vars, query_imports, {"close_var"}, chunk_imports, {}
        )
        score_far_matched = weighted_ast_attention_score(
            query_vars, query_imports, {"far_var"}, chunk_imports, {}
        )
        # matcher la variable à forte attention doit donner un meilleur score
        self.assertGreater(score_close_matched, score_far_matched)

    def test_import_weight_multiplier_exceeds_variable_weight_multiplier(self):
        # à attention_score=1.0 et doc_weight=1.0, un import matché (2.5) doit peser
        # plus qu'une variable matchée (2.0) dans un score partiel
        query_vars = {"v": 1.0}
        query_imports = {"imp"}
        doc_weights = {}

        only_var_matched = weighted_ast_attention_score(query_vars, query_imports, {"v"}, set(), doc_weights)
        only_import_matched = weighted_ast_attention_score(query_vars, query_imports, set(), {"imp"}, doc_weights)
        self.assertGreater(only_import_matched, only_var_matched)

    def test_realistic_attention_score_from_ast_distance(self):
        # attention_score = exp(-lambda * distance_ast), calculé par l'appelant
        lam = 0.5
        distance = 2
        attention_score = math.exp(-lam * distance)
        query_vars = {"trainer": attention_score}
        score = weighted_ast_attention_score(query_vars, set(), {"trainer"}, set(), {})
        self.assertEqual(score, 1.0)

    def test_score_always_within_bounds(self):
        query_vars = {"a": 0.3, "b": 0.7, "c": 1.0}
        query_imports = {"x", "y", "z"}
        doc_weights = {"a": 2.0, "b": 0.5, "x": 3.0}
        chunk_symbol_variants = (set(), {"a"}, {"a", "b"}, {"a", "b", "c"}, {"a", "b", "c", "noise"})
        chunk_import_variants = (set(), {"x"}, {"x", "y"}, {"x", "y", "z"}, {"x", "y", "z", "extra"})
        for chunk_symbols in chunk_symbol_variants:
            for chunk_imports in chunk_import_variants:
                score = weighted_ast_attention_score(
                    query_vars, query_imports, chunk_symbols, chunk_imports, doc_weights
                )
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_runs_well_under_one_millisecond(self):
        # taille réaliste (large) : 50 variables + 20 imports côté requête,
        # chunk avec 200 symboles + 30 imports.
        query_vars = {f"var_{i}": (i % 10) / 10 for i in range(50)}
        query_imports = {f"import_{i}" for i in range(20)}
        chunk_symbols = {f"var_{i}" for i in range(0, 200, 2)}
        chunk_imports = {f"import_{i}" for i in range(0, 30, 3)}
        doc_weights = {f"var_{i}": 1.5 for i in range(50)}

        # chauffe (évite de mesurer un éventuel coût d'import/JIT froid)
        weighted_ast_attention_score(query_vars, query_imports, chunk_symbols, chunk_imports, doc_weights)

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            weighted_ast_attention_score(query_vars, query_imports, chunk_symbols, chunk_imports, doc_weights)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations

        self.assertLess(elapsed_ms, 1.0, f"temps moyen par appel: {elapsed_ms:.4f} ms")


if __name__ == "__main__":
    unittest.main()
