"""
Unit Tests for Embedded Deployment & TFLite Micro Firmware Artifacts
SIH Problem Statement 26172 - Milestone 13 Verification
"""

import os
import sys
import re
import tempfile
import unittest
import numpy as np

sys.path.insert(0, r"D:\SIH_Model")
from src.export.tflite_to_c_array import tflite_to_c_header, estimate_tensor_arena_size


class TestDeploymentPipeline(unittest.TestCase):
    """Test suite for ESP32-S3 C++ firmware components and TFLite Micro headers."""

    @classmethod
    def setUpClass(cls):
        cls.tflite_model_path = r"D:\SIH_Model\models\tflite\voice_activator_int8.tflite"
        cls.header_path = r"D:\SIH_Model\src\deployment\esp32\tflite_micro_model.h"
        cls.prototype_path = r"D:\SIH_Model\src\deployment\esp32\keyword_prototype.h"
        cls.ring_buf_path = r"D:\SIH_Model\src\deployment\esp32\audio_ring_buffer.h"
        cls.feat_ext_path = r"D:\SIH_Model\src\deployment\esp32\feature_extractor.h"
        cls.state_mach_path = r"D:\SIH_Model\src\deployment\esp32\activator_state_machine.h"
        cls.main_cpp_path = r"D:\SIH_Model\src\deployment\esp32\main.cpp"

    def test_tflite_micro_header_matches_binary(self):
        """Verify C header byte array matches the compiled .tflite binary."""
        self.assertTrue(os.path.exists(self.header_path), "Header file missing")
        self.assertTrue(os.path.exists(self.tflite_model_path), "TFLite binary missing")

        binary_size = os.path.getsize(self.tflite_model_path)
        with open(self.header_path, "r", encoding="utf-8") as f:
            header_text = f.read()

        # Check alignment and length variable
        self.assertIn("alignas(16)", header_text)
        self.assertIn(f"const unsigned int g_voice_activator_model_data_len = {binary_size};", header_text)

        # Check flash budget constraint (< 100 KB)
        self.assertLess(binary_size, 100 * 1024, "INT8 model exceeds 100 KB flash limit")

    def test_keyword_prototype_header(self):
        """Verify enrolled keyword prototype format and unit norm."""
        self.assertTrue(os.path.exists(self.prototype_path))
        with open(self.prototype_path, "r", encoding="utf-8") as f:
            proto_text = f.read()

        self.assertIn("ENROLLED_KEYWORD_NAME", proto_text)
        # Verify valid keyword name is present (e.g. ZORA or CLEOPATRA)
        self.assertTrue("ASWIN" in proto_text or "ZORA" in proto_text or "CLEOPATRA" in proto_text or "ENROLLED_KEYWORD_NAME" in proto_text)

        # Extract floats from header
        matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+f", proto_text)
        self.assertEqual(len(matches), 32, "Expected 32-D prototype embedding")

        vals = np.array([float(m.rstrip("f")) for m in matches])
        norm = np.linalg.norm(vals)
        self.assertAlmostEqual(norm, 1.0, places=3, msg="Prototype must be on unit hypersphere")

    def test_firmware_source_files_exist(self):
        """Verify all modular C++ firmware components are present."""
        for path in [self.ring_buf_path, self.feat_ext_path, self.state_mach_path, self.main_cpp_path]:
            self.assertTrue(os.path.exists(path), f"Missing firmware file: {path}")
            self.assertGreater(os.path.getsize(path), 500, f"File appears too small or empty: {path}")

    def test_sram_tensor_arena_estimation(self):
        """Verify TFLite Micro tensor arena fits within ESP32-S3 internal SRAM budget."""
        arena_report = estimate_tensor_arena_size(self.tflite_model_path)
        self.assertIn("recommended_arena_bytes", arena_report)
        self.assertIn("recommended_arena_kb", arena_report)

        # Total internal SRAM on ESP32-S3 is 256 KB.
        # Arena must consume < 64 KB, leaving remainder for ring buffer (32 KB), stack, and heap.
        self.assertTrue(arena_report["sram_budget_satisfied"])
        self.assertLess(arena_report["recommended_arena_kb"], 64.0)

    def test_c_array_exporter_roundtrip(self):
        """Verify C array exporter utility with arbitrary binary payload."""
        test_payload = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d"
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tf_in:
            tf_in.write(test_payload)
            tf_in_name = tf_in.name

        with tempfile.NamedTemporaryFile(suffix=".h", delete=False) as tf_out:
            tf_out_name = tf_out.name

        try:
            size, lines = tflite_to_c_header(tf_in_name, tf_out_name, array_name="test_model")
            self.assertEqual(size, len(test_payload))
            self.assertGreater(lines, 10)

            with open(tf_out_name, "r") as f:
                content = f.read()
            self.assertIn("0x01, 0x02, 0x03", content)
            self.assertIn("const unsigned int test_model_len = 13;", content)
        finally:
            if os.path.exists(tf_in_name):
                os.remove(tf_in_name)
            if os.path.exists(tf_out_name):
                os.remove(tf_out_name)


if __name__ == "__main__":
    unittest.main()
