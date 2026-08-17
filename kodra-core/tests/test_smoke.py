import unittest
import os
import sys

SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from scripts.verify_project import verify

class TestSmoke(unittest.TestCase):
    def test_full_pipeline_verification(self):
        success = verify()
        self.assertTrue(success)

if __name__ == "__main__":
    unittest.main()
