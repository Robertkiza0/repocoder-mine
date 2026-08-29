import os
from typing import List, Set, Dict
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# ==========================================
# 1. ANALYSTE AST (Tree-Sitter Extractor)
# ==========================================
class ASTContextAnalyzer:
    def __init__(self):
        self.py_language = Language(tspython.language())
        self.parser = Parser(self.py_language)

    def extract_context(self, code_snippet: str) -> Dict[str, Set[str]]:
        """
        Extrait en 0 ms les variables locales et les imports du code source.
        """
        if not code_snippet.strip():
            return {"variables": set(), "imports": set()}

        tree = self.parser.parse(bytes(code_snippet, "utf8"))
        variables = set()
        imports = set()

        stack = [tree.root_node]
        while stack:
            node = stack.pop()

            # Extraction des variables (Assignments)
            if node.type == "assignment":
                left_node = node.child_by_field_name("left")
                if left_node and left_node.type == "identifier":
                    var_name = code_snippet[left_node.start_byte:left_node.end_byte]
                    variables.add(var_name)

            # Extraction des Imports (import X / from X import Y)
            elif node.type in ("import_statement", "import_from_statement"):
                import_text = code_snippet[node.start_byte:node.end_byte].strip()
                imports.add(import_text)

            stack.extend(node.children)

        return {"variables": variables, "imports": imports}


# ==========================================
# 2. AGENT LLM MONO-TOUR (Single-Turn Agent)
# ==========================================
class SingleTurnCodeAgent:
    SYSTEM_PROMPT = (
        "Vous êtes un assistant de complétion de code en temps réel pour IDE.\n"
        "Analyse le code incomplet et utilise les exemples du dépôt pour prédire la suite.\n"
        "Règles strictes :\n"
        "1. Utilise en priorité les variables locales actives et les fonctions importées.\n"
        "2. N'invente pas de nouvelles signatures si des fonctions équivalentes existent dans les exemples.\n"
        "3. Génère UNIQUEMENT le code de complétion venant immédiatement après le curseur."
    )

    def __init__(self, llm_client, model_name: str = "qwen2.5-coder-7b"):
        """
        llm_client: Le client LLM (ex: OpenAI, Ollama, vLLM, HuggingFace)
        """
        self.client = llm_client
        self.model_name = model_name
        self.ast_analyzer = ASTContextAnalyzer()

    def build_structured_prompt(
        self,
        target_file_path: str,
        unfinished_code: str,
        retrieved_chunks: List[Dict[str, str]]
    ) -> str:
        """
        Construit le prompt enrichi avec les métadonnées de l'AST (Path, Imports, Variables)
        """
        # Analyse AST en temps réel sur le code incomplet
        ast_metadata = self.ast_analyzer.extract_context(unfinished_code)

        vars_str = ", ".join(ast_metadata["variables"]) if ast_metadata["variables"] else "None"
        imports_str = "\n".join(ast_metadata["imports"]) if ast_metadata["imports"] else "None"

        # Formate les exemples du dépôt récupérés par le Retriever Sparse
        retrieved_context_str = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            retrieved_context_str += f"\n# Example {i} | File: {chunk.get('file_path', 'unknown')}\n"
            retrieved_context_str += f"{chunk.get('raw_code', '')}\n"

        prompt = f"""=================== METADATA & CONTEXT ===================
FILE PATH: {target_file_path}

IMPORTS:
{imports_str}

ACTIVE SCOPE VARIABLES:
{vars_str}

=================== RETRIEVED REPO EXAMPLES =============
{retrieved_context_str}
=================== TARGET CODE TO COMPLETE =============
{unfinished_code}"""

        return prompt

    def generate_completion(
        self,
        target_file_path: str,
        unfinished_code: str,
        retrieved_chunks: List[Dict[str, str]]
    ) -> str:
        """
        Exécute la complétion en un seul tour (Single-Turn).
        """
        structured_prompt = self.build_structured_prompt(
            target_file_path, unfinished_code, retrieved_chunks
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": structured_prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )

        return response.choices[0].message.content
