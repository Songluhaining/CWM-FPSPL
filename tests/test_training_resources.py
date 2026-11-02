"""Resource recording must not depend on a Windows-specific Python installation."""
import unittest
from unittest.mock import patch
from cwm_fp.models.package_training import process_memory


class TrainingResourceTests(unittest.TestCase):
    def test_actual_process_has_positive_peak(self):
        info=process_memory()
        self.assertIsNotNone(info['peak_working_set_bytes'])
        self.assertGreater(info['peak_working_set_bytes'],0)

    def test_unknown_measurement_is_null_with_reason(self):
        with patch('builtins.__import__',side_effect=ImportError('deliberately unavailable')):
            info=process_memory()
        self.assertIsNone(info['peak_working_set_bytes'])
        self.assertIsNone(info['final_RSS_bytes'])
        self.assertIn('deliberately unavailable',info['memory_measurement_reason'])
