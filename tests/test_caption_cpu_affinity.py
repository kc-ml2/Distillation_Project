import os
import unittest

from data.eval_validation_caption import parse_cpu_list, cpu_affinity


class ParseCpuListTest(unittest.TestCase):
    def test_range_and_singles(self):
        self.assertEqual(parse_cpu_list("0-3"), {0, 1, 2, 3})
        self.assertEqual(parse_cpu_list("0-1,4,6-7"), {0, 1, 4, 6, 7})

    def test_empty_is_none(self):
        self.assertIsNone(parse_cpu_list(None))
        self.assertIsNone(parse_cpu_list(""))


class CpuAffinityTest(unittest.TestCase):
    def test_restores_original(self):
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("no sched_getaffinity on this platform")
        orig = os.sched_getaffinity(0)
        with cpu_affinity({0}):
            self.assertEqual(os.sched_getaffinity(0), {0})
        self.assertEqual(os.sched_getaffinity(0), orig)

    def test_none_is_noop(self):
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("no sched_getaffinity on this platform")
        orig = os.sched_getaffinity(0)
        with cpu_affinity(None):
            self.assertEqual(os.sched_getaffinity(0), orig)
        self.assertEqual(os.sched_getaffinity(0), orig)


if __name__ == "__main__":
    unittest.main()
