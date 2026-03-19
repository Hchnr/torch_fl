"""
torch_fl CUDA 透传路径冒烟测试。
验证 import fl 在 CUDA 模式下的基本功能。

运行方式:
  FL_BACKEND=cuda FL_GEMS=0 python -m pytest tests/test_cuda_smoke.py -v
  或:
  FL_BACKEND=cuda FL_GEMS=0 python tests/test_cuda_smoke.py
"""
import os

os.environ["FL_BACKEND"] = "cuda"
os.environ["FL_GEMS"] = "0"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402
import unittest  # noqa: E402

CUDA_AVAILABLE = torch.cuda.is_available()


class TestDetect(unittest.TestCase):
    """测试硬件检测模块"""

    def test_detect_returns_cuda_with_env(self):
        from fl._detect import detect_backend
        old = os.environ.get("FL_BACKEND")
        os.environ["FL_BACKEND"] = "cuda"
        try:
            self.assertEqual(detect_backend(), "cuda")
        finally:
            if old is not None:
                os.environ["FL_BACKEND"] = old

    def test_detect_returns_npu_with_env(self):
        from fl._detect import detect_backend
        os.environ["FL_BACKEND"] = "npu"
        try:
            self.assertEqual(detect_backend(), "npu")
        finally:
            os.environ["FL_BACKEND"] = "cuda"

    def test_detect_returns_mlu_with_env(self):
        from fl._detect import detect_backend
        os.environ["FL_BACKEND"] = "mlu"
        try:
            self.assertEqual(detect_backend(), "mlu")
        finally:
            os.environ["FL_BACKEND"] = "cuda"


class TestCUDAImport(unittest.TestCase):
    """测试 CUDA 透传模式下的 import 和基本 API"""

    def test_import_fl(self):
        import fl
        self.assertEqual(fl.BACKEND_NAME, "cuda")

    def test_C_is_none(self):
        import fl
        self.assertIsNone(fl._C)

    def test_backend_config(self):
        import fl
        self.assertEqual(fl._backend_config["backend_name"], "cuda")
        self.assertTrue(fl._backend_config["is_native"])
        self.assertEqual(fl._backend_config["comm_backend"], "nccl")

    def test_device_returns_cuda(self):
        import fl
        import torch
        d = fl.device(0)
        self.assertEqual(d, torch.device("cuda", 0))

    def test_dist_backend(self):
        import fl
        self.assertEqual(fl.dist_backend(), "nccl")


class TestCUDAPassthrough(unittest.TestCase):
    """测试 CUDA 透传 API 是否正确映射到 torch.cuda"""

    def test_is_available_matches_torch(self):
        import fl
        import torch
        self.assertEqual(fl.is_available(), torch.cuda.is_available())

    def test_device_count_matches_torch(self):
        import fl
        import torch
        self.assertEqual(fl.device_count(), torch.cuda.device_count())

    @unittest.skipUnless(
        CUDA_AVAILABLE, "CUDA not available"
    )
    def test_current_device_matches_torch(self):
        import fl
        import torch
        self.assertEqual(fl.current_device(), torch.cuda.current_device())

    @unittest.skipUnless(
        CUDA_AVAILABLE, "CUDA not available"
    )
    def test_get_device_name(self):
        import fl
        name = fl.get_device_name(0)
        self.assertIsInstance(name, str)
        self.assertTrue(len(name) > 0)

    @unittest.skipUnless(
        CUDA_AVAILABLE, "CUDA not available"
    )
    def test_stream_type(self):
        import fl
        import torch
        self.assertIs(fl.Stream, torch.cuda.Stream)
        self.assertIs(fl.Event, torch.cuda.Event)

    @unittest.skipUnless(
        CUDA_AVAILABLE, "CUDA not available"
    )
    def test_memory_allocated(self):
        import fl
        mem = fl.memory_allocated(0)
        self.assertIsInstance(mem, int)
        self.assertGreaterEqual(mem, 0)


class TestCUDACompute(unittest.TestCase):
    """测试 CUDA 模式下的实际计算"""

    @unittest.skipUnless(
        CUDA_AVAILABLE, "CUDA not available"
    )
    def test_tensor_on_device(self):
        import fl
        import torch
        x = torch.randn(3, 3, device=fl.device(0))
        self.assertTrue(x.is_cuda)
        self.assertEqual(x.device, torch.device("cuda", 0))

    @unittest.skipUnless(
        CUDA_AVAILABLE, "CUDA not available"
    )
    def test_matmul(self):
        import fl
        import torch
        x = torch.randn(4, 4, device=fl.device(0))
        y = torch.randn(4, 4, device=fl.device(0))
        z = x @ y
        self.assertEqual(z.shape, (4, 4))
        self.assertTrue(z.is_cuda)

    @unittest.skipUnless(
        CUDA_AVAILABLE, "CUDA not available"
    )
    def test_synchronize(self):
        import fl
        import torch
        x = torch.randn(100, 100, device=fl.device(0))
        _ = x @ x
        fl.synchronize()  # 不应抛异常


class TestGemsModule(unittest.TestCase):
    """测试 FlagGems 模块（禁用状态）"""

    def test_gems_not_enabled(self):
        """FL_GEMS=0 时 gems 不应该被启用"""
        import fl
        self.assertFalse(fl.gems.is_enabled())

    def test_gems_registered_ops_empty(self):
        import fl
        self.assertEqual(fl.gems.registered_ops(), [])

    def test_gems_is_available_returns_bool(self):
        import fl
        result = fl.gems.is_available()
        self.assertIsInstance(result, bool)


class TestBackendConfig(unittest.TestCase):
    """测试后端配置接口"""

    def test_cuda_config(self):
        from fl.backends.cuda import get_config
        cfg = get_config()
        self.assertEqual(cfg["backend_name"], "cuda")
        self.assertTrue(cfg["is_native"])
        self.assertEqual(cfg["gems_config"]["dispatch_key"], "CUDA")

    def test_npu_config(self):
        from fl.backends.npu import get_config
        cfg = get_config()
        self.assertEqual(cfg["backend_name"], "npu")
        self.assertFalse(cfg["is_native"])
        self.assertEqual(cfg["gems_config"]["dispatch_key"], "PrivateUse1")

    def test_mlu_config(self):
        from fl.backends.mlu import get_config
        cfg = get_config()
        self.assertEqual(cfg["backend_name"], "mlu")
        self.assertFalse(cfg["is_native"])
        self.assertEqual(cfg["comm_backend"], "cncl")


if __name__ == "__main__":
    unittest.main()
