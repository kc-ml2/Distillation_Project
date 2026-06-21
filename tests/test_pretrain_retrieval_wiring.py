import inspect
import unittest

import pretrain


class PretrainRetrievalWiringTest(unittest.TestCase):
    def test_train_accepts_model_without_ddp_and_retrieval_val_runner(self):
        sig = inspect.signature(pretrain.train)
        self.assertIn("model_without_ddp", sig.parameters)
        self.assertIn("retrieval_val_runner", sig.parameters)
        self.assertIsNone(sig.parameters["model_without_ddp"].default)
        self.assertIsNone(sig.parameters["retrieval_val_runner"].default)

    def test_train_body_calls_val_retrieval_during_train(self):
        source = inspect.getsource(pretrain.train)
        self.assertIn("retrieval_val_runner.val_retrieval_during_train(", source)

    def test_main_body_builds_and_calls_runner(self):
        source = inspect.getsource(pretrain.main)
        self.assertIn("eval_validation_retrieval.build_pretrain_retrieval_val_runner(", source)
        self.assertIn("retrieval_val_runner.run_epoch_end(", source)


if __name__ == "__main__":
    unittest.main()
