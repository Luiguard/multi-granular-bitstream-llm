#!/usr/bin/env python3
"""
Test Suite for Universal Standalone AI Package Generator (All-in-One Deployment Bundle).
"""

import ast
import os
import sys
import zipfile
import unittest

sys.path.insert(0, "/home/benjamin/Bilder")

from pipeline.model_builder_engine import ModelArchitectureSpecs


class TestCompleteBundle(unittest.TestCase):

    def setUp(self):
        self.temp_zip = "/home/benjamin/Bilder/data/test_scratch/test_complete_bundle.zip"
        os.makedirs(os.path.dirname(self.temp_zip), exist_ok=True)
        if os.path.exists(self.temp_zip):
            os.remove(self.temp_zip)

        self.specs = ModelArchitectureSpecs(
            name="Bitstream-Universal-Edge",
            d_model=512,
            n_layers=8,
            n_heads=8,
            num_experts=4,
            top_k=1,
            ffn_multiplier=2.0,
            vocab_size=262144,
            rank_embedding=32,
            max_seq_len=4096,
        )

    def tearDown(self):
        if os.path.exists(self.temp_zip):
            os.remove(self.temp_zip)

    def test_python_scripts_syntax(self):
        """Verifies that all generated scripts (model, train, api_server, web_chat, cli_chat) are valid Python syntax."""
        scripts = {
            "model.py": self.specs.generate_pytorch_code(),
            "train.py": self.specs.generate_training_script(),
            "api_server.py": self.specs.generate_api_server_script(),
            "web_chat.py": self.specs.generate_web_chat_script(),
            "cli_chat.py": self.specs.generate_cli_chat_script(),
        }

        for filename, code in scripts.items():
            try:
                tree = ast.parse(code)
                self.assertIsNotNone(tree, f"{filename} syntax parsing failed")
            except SyntaxError as e:
                self.fail(f"Syntax error in generated {filename}: {e}")

    def test_zip_bundle_structure(self):
        """Verifies that create_training_bundle_zip generates a complete, self-contained archive."""
        zip_path = self.specs.create_training_bundle_zip(self.temp_zip)
        self.assertTrue(os.path.exists(zip_path))

        expected_files = [
            "model.py",
            "train.py",
            "api_server.py",
            "web_chat.py",
            "cli_chat.py",
            "config.json",
            "start_linux.sh",
            "start_windows.bat",
            "start_mac.sh",
            "Dockerfile",
            "docker-compose.yml",
            "requirements.txt",
            "README.md",
        ]

        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            for ef in expected_files:
                self.assertIn(ef, namelist, f"Missing file in bundle zip: {ef}")

            # Verify README contains correct specs
            readme_text = zf.read("README.md").decode("utf-8")
            self.assertIn("Bitstream-Universal-Edge", readme_text)
            self.assertIn("curl http://localhost:8000/v1/chat/completions", readme_text)


if __name__ == "__main__":
    unittest.main()
