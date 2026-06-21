import importlib
import unittest


class EvalPretrainRetrievalScriptTest(unittest.TestCase):
    def test_module_imports_cleanly(self):
        importlib.import_module("eval_pretrain_retrieval")

    def test_parser_requires_checkpoint_and_defaults_split_to_val(self):
        module = importlib.import_module("eval_pretrain_retrieval")
        parser = module.build_arg_parser()

        args = parser.parse_args(["--checkpoint", "/tmp/fake.pth"])
        self.assertEqual(args.checkpoint, "/tmp/fake.pth")
        self.assertEqual(args.split, "val")

        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_parser_accepts_test_split(self):
        module = importlib.import_module("eval_pretrain_retrieval")
        parser = module.build_arg_parser()
        args = parser.parse_args(["--checkpoint", "/tmp/fake.pth", "--split", "test"])
        self.assertEqual(args.split, "test")


if __name__ == "__main__":
    unittest.main()
