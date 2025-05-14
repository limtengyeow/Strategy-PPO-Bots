# import unittest
import json
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

class TestTrainPPO(unittest.TestCase):
    def test_config_structure(self):
        with open(os.path.join(project_root, "config.json")) as f:
            cfg = json.load(f)
        self.assertIn("training", cfg)
