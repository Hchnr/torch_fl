# Owner(s): ["module: PrivateUse1"]

"""
Test Qwen3-0.6B inference on the openreg device.

The openreg device is a CPU-emulated PrivateUse1 backend. Operations not
natively registered will automatically fall back to CPU, so a full
transformer model can run on it end-to-end.

Usage:
    cd /tmp
    PYTHONPATH=<path-to-torch_openreg> python test_qwen3_inference.py
"""

import unittest

import torch
import torch_openreg  # noqa: F401  — registers the "openreg" backend
from torch.testing._internal.common_utils import run_tests, TestCase


def _has_transformers():
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


MODEL_ID = "Qwen/Qwen3-0.6B"


@unittest.skipUnless(_has_transformers(), "requires the `transformers` library")
class TestQwen3Inference(TestCase):
    """Verify that Qwen3-0.6B can be loaded, moved to the openreg device,
    and produce valid inference results."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        cls.model.to("openreg")
        cls.model.eval()

    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    def test_model_device(self):
        """All parameters should reside on the openreg device."""
        for name, param in self.model.named_parameters():
            self.assertEqual(
                param.device.type,
                "openreg",
                f"Parameter {name} is on {param.device}, expected openreg",
            )

    def test_model_dtype(self):
        """Parameters should keep the requested dtype."""
        for name, param in self.model.named_parameters():
            self.assertIn(
                param.dtype,
                (torch.float32, torch.float64),
                f"Parameter {name} has unexpected dtype {param.dtype}",
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def test_forward_pass(self):
        """A single forward pass should return logits of the correct shape."""
        prompt = "Hello"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("openreg")
        attention_mask = inputs["attention_mask"].to("openreg")

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        batch_size, seq_len = input_ids.shape
        vocab_size = self.model.config.vocab_size

        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[1], seq_len)
        self.assertEqual(logits.shape[2], vocab_size)
        self.assertTrue(
            torch.isfinite(logits.cpu()).all(), "Logits contain non-finite values"
        )

    def test_forward_batch(self):
        """Forward pass with a batch of inputs."""
        prompts = ["Hi", "How are you?"]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to("openreg")
        attention_mask = inputs["attention_mask"].to("openreg")

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs.logits.shape[0], len(prompts))
        self.assertTrue(torch.isfinite(outputs.logits.cpu()).all())

    # ------------------------------------------------------------------
    # Token generation (greedy, step-by-step to avoid device mismatch
    # inside transformers' generate() internals)
    # ------------------------------------------------------------------
    def test_greedy_decode(self):
        """Manual greedy decoding loop to verify token generation works."""
        prompt = "The capital of France is"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("openreg")

        max_new_tokens = 20
        generated = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids=generated)
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token_id], dim=-1)
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        self.assertGreater(generated.shape[1], input_ids.shape[1])
        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded), len(prompt))

    def test_sampling_decode(self):
        """Manual sampling-based decoding loop."""
        prompt = "Once upon a time"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("openreg")

        max_new_tokens = 10
        temperature = 0.7
        generated = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids=generated)
                logits = outputs.logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token_id], dim=-1)
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        self.assertGreater(generated.shape[1], input_ids.shape[1])

    # ------------------------------------------------------------------
    # Cross-device consistency
    # ------------------------------------------------------------------
    def test_weight_roundtrip(self):
        """Weights survive a cpu -> openreg -> cpu round-trip intact."""
        # Capture a reference parameter on openreg
        param_name = next(iter(dict(self.model.named_parameters())))
        ref = dict(self.model.named_parameters())[param_name].cpu().clone()

        # Round-trip: openreg -> cpu -> openreg
        self.model.to("cpu")
        cpu_val = dict(self.model.named_parameters())[param_name].clone()
        self.model.to("openreg")

        self.assertEqual(ref, cpu_val, f"Weight {param_name} changed after round-trip")

    def test_forward_deterministic(self):
        """Two forward passes with the same input produce identical logits."""
        prompt = "PyTorch is"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs_openreg = {k: v.to("openreg") for k, v in inputs.items()}

        with torch.no_grad():
            logits1 = self.model(**inputs_openreg).logits.cpu()
            logits2 = self.model(**inputs_openreg).logits.cpu()

        self.assertTrue(
            torch.equal(logits1, logits2),
            "Two identical forward passes produced different logits",
        )

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------
    def test_model_to_cpu_and_back(self):
        """Model can be moved to CPU and back to openreg."""
        self.model.to("cpu")
        for name, param in self.model.named_parameters():
            self.assertTrue(param.is_cpu, f"{name} not on CPU after .to('cpu')")

        self.model.to("openreg")
        for name, param in self.model.named_parameters():
            self.assertEqual(
                param.device.type,
                "openreg",
                f"{name} not on openreg after .to('openreg')",
            )


if __name__ == "__main__":
    run_tests()
