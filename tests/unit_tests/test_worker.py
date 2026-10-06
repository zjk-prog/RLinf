from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import logging

# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pickle
import sys
import tempfile
import time
import types
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock
from unittest.mock import patch

import pytest
import ray
from omegaconf import OmegaConf

import rlinf.scheduler.hardware.accelerators.nvidia_gpu as nv_module
from rlinf.scheduler import (
    Cluster,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Tracer,
    Worker,
    WorkerAddress,
)
from rlinf.scheduler.cluster.utils import DistributedRayLogCollector
from rlinf.scheduler.hardware.accelerators.intel_gpu import IntelGPUManager
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager
from rlinf.scheduler.manager.coll_manager import CollectiveManager
from rlinf.scheduler.manager.manager import Manager


def accelerator_is_available():
    """Return whether the Worker accelerator backend is available."""
    return (
        Worker.torch_platform is not None
        and hasattr(Worker.torch_platform, "is_available")
        and Worker.torch_platform.is_available()
    )


# Fixture to provide a ClusterResource instance for the test session
@pytest.fixture(scope="module")
def cluster():
    """Provides a ClusterResource instance for the tests."""
    # Use a small, fixed number of GPUs for consistent testing
    return Cluster(num_nodes=1)


# A basic Worker class for testing purposes
class BasicTestWorker(Worker):
    """A simple Worker implementation for testing basic functionality."""

    def __init__(self, arg1=None):
        super().__init__()
        self.arg1 = arg1
        self.initialized = True

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size

    def get_init_arg(self):
        return self.arg1


# A WorkerGroup-decorated class for testing distributed functionality
class DistributedTestWorker(Worker):
    """A WorkerGroup for testing distributed operations."""

    def __init__(self):
        super().__init__()

    def get_env_info(self):
        """Returns a dictionary of environment information for the worker."""
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "node_id": self._cluster_node_rank,
            "gpu_id": self._local_accelerator_rank,
            "node_local_rank": self._node_local_rank,
        }

    def sum_with_rank(self, value):
        """Adds the worker's rank to the given value."""
        return value + self._rank


class TestClusterResource:
    """Tests for the ClusterResource class."""

    def test_cluster_initialization(self, cluster: Cluster):
        """Verify that the cluster is initialized with correct properties."""
        assert cluster._num_nodes == 1
        if accelerator_is_available():
            assert cluster.num_accelerators >= 1


class TestWorkerAddress:
    """Tests for the WorkerAddress class."""

    def test_worker_address_naming(self):
        """Verify that WorkerAddress generates correct names."""
        addr = WorkerAddress("MyWorkerGroup", 5)
        assert addr.root_group_name == "MyWorkerGroup"
        assert addr.rank == 5
        assert addr.get_name() == "MyWorkerGroup:5"


class StaticMethodWorker(Worker):
    """A Worker exposing a public staticmethod, wrapped by WorkerMeta."""

    @staticmethod
    def add(first: int, second: int) -> int:
        return first + second


class TestWorkerMeta:
    """Tests for the WorkerMeta method wrapping."""

    def test_staticmethod_stays_a_staticmethod(self):
        """Verify WorkerMeta does not turn a staticmethod into an instance method."""
        assert isinstance(StaticMethodWorker.__dict__["add"], staticmethod)

    def test_staticmethod_call_does_not_rebind_self(self):
        """Verify calling a staticmethod through an instance passes no self."""
        worker = object.__new__(StaticMethodWorker)
        assert worker.add(1, 2) == 3
        assert StaticMethodWorker.add(1, 2) == 3


class TestManagerNamespace:
    """Tests for manager namespace propagation."""

    def test_manager_runtime_env_vars_include_cluster_namespace(self):
        """Verify manager runtime env always includes the cluster namespace."""
        with mock.patch.object(Cluster, "NAMESPACE", "test-namespace"):
            runtime_env = Manager.get_runtime_env_vars()

        assert runtime_env["CLUSTER_NAMESPACE"] == "test-namespace"

    def test_sync_cluster_namespace_from_env(self):
        """Verify manager syncs the cluster namespace from its runtime env."""
        with mock.patch.object(Cluster, "NAMESPACE", "original-namespace"):
            with mock.patch.dict(
                os.environ, {"CLUSTER_NAMESPACE": "env-namespace"}, clear=False
            ):
                CollectiveManager()
                assert Cluster.NAMESPACE == "env-namespace"


class TestWorkerGroup:
    """Tests for the WorkerGroup class and its interactions."""

    def test_worker_group_creation(self, cluster: Cluster):
        """Verify that a WorkerGroup can be created successfully."""
        if accelerator_is_available():
            num_workers = cluster.num_accelerators
        else:
            num_workers = 1
        worker_group = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="dist_test_1"
        )

        # Check that the correct number of actors were created
        assert len(worker_group.worker_info_list) == num_workers

        # Verify that we can get results from the workers
        results = worker_group.get_env_info().wait()
        assert len(results) == num_workers
        ranks = sorted([info["rank"] for info in results])
        assert ranks == list(range(num_workers))

    def test_execute_on_all_workers(self, cluster: Cluster):
        """Test calling a method on all workers in a group."""
        if accelerator_is_available():
            num_workers = cluster.num_accelerators
        else:
            num_workers = 1
        worker_group = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="dist_test_2"
        )

        base_value = 10
        results = worker_group.sum_with_rank(base_value).wait()

        assert len(results) == num_workers
        expected_results = sorted([base_value + i for i in range(num_workers)])
        assert sorted(results) == expected_results

    def test_execute_on_specific_ranks(self, cluster: Cluster):
        """Test calling a method on a subset of workers in a group."""
        if accelerator_is_available():
            placement = PackedPlacementStrategy(0, cluster.num_accelerators - 1)
        else:
            placement = NodePlacementStrategy([0] * 8)
        worker_group = DistributedTestWorker.create_group().launch(
            cluster=cluster, placement_strategy=placement, name="dist_test_3"
        )

        target_ranks = (0, 1)
        base_value = 20
        results = (
            worker_group.execute_on(*target_ranks).sum_with_rank(base_value).wait()
        )

        assert len(results) == len(target_ranks)
        expected_results = sorted([base_value + rank for rank in target_ranks])
        assert sorted(results) == expected_results

    def test_multiple_worker_groups(self, cluster: Cluster):
        """Test the creation and operation of multiple independent worker groups."""
        if accelerator_is_available():
            num_workers = cluster.num_accelerators
        else:
            num_workers = 1
        group1 = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="multi_group_1"
        )
        group2 = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="multi_group_2"
        )

        # Call a method on group 1
        results1 = group1.sum_with_rank(100).wait()
        assert len(results1) == num_workers
        assert sorted(results1) == [100 + i for i in range(num_workers)]

        # Call a method on group 2
        results2 = group2.sum_with_rank(200).wait()
        assert len(results2) == num_workers
        assert sorted(results2) == [200 + i for i in range(num_workers)]


class TestLoadUserExtensions:
    """Tests for the Worker._load_user_extensions method."""

    def _create_mock_worker(self):
        """Create a minimal mock worker instance for testing _load_user_extensions."""
        worker = object.__new__(Worker)
        return worker

    def test_no_action_when_env_var_not_set(self):
        """Verify no action is taken when RLINF_EXT_MODULE is not set."""
        worker = self._create_mock_worker()
        os.environ.pop("RLINF_EXT_MODULE", None)

        with mock.patch("importlib.import_module") as mock_import:
            worker._load_user_extensions()
            mock_import.assert_not_called()

    def test_extension_module_loaded_and_register_called(self):
        """Verify extension module is loaded and register() is called."""
        worker = self._create_mock_worker()
        mock_module = types.ModuleType("mock_ext_module")
        mock_module.register = mock.Mock()

        with mock.patch.dict(os.environ, {"RLINF_EXT_MODULE": "mock_ext_module"}):
            with mock.patch("importlib.import_module", return_value=mock_module):
                worker._load_user_extensions()
                mock_module.register.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", __file__])


class LocalRayLogCollector(DistributedRayLogCollector):
    """Collector variant using local temp directories for testing."""

    def __init__(self, *args, logs_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self._logs_dir = logs_dir

    def _get_ray_logs_dir(self) -> Path:
        return self._logs_dir

    def _resolve_registered_workers(self, logs_dir: Path) -> None:
        # Keep worker-to-log mapping fully controlled by each test.
        return


def _new_collector(logs_dir: Path, output_dir: Path) -> LocalRayLogCollector:
    logger = logging.getLogger("test.distributed_log_collector")
    return LocalRayLogCollector(
        logger=logger,
        output_dir=output_dir,
        logs_dir=logs_dir,
        poll_interval_s=0.1,
    )


def test_collector_is_pickleable_even_after_start(tmp_path: Path):
    logs_dir = tmp_path / "ray_logs"
    output_dir = tmp_path / "split_logs"
    logs_dir.mkdir(parents=True)

    collector = _new_collector(logs_dir=logs_dir, output_dir=output_dir)
    assert collector.start() is True
    try:
        payload = pickle.dumps(collector)
        restored = pickle.loads(payload)
    finally:
        collector.stop()

    assert isinstance(restored, LocalRayLogCollector)
    assert restored._thread is None
    assert restored._started is False


def test_process_once_reads_from_start_and_appends_incrementally(tmp_path: Path):
    logs_dir = tmp_path / "ray_logs"
    output_dir = tmp_path / "split_logs"
    logs_dir.mkdir(parents=True)

    worker_log = logs_dir / "worker-w1-j1-123.out"
    worker_log.write_text("line-1\nline-2\n", encoding="utf-8")

    collector = _new_collector(logs_dir=logs_dir, output_dir=output_dir)
    collector._log_file_map[worker_log] = ("actor_group:0", "0")

    collector._process_once(logs_dir, new_file_offset_from_start=True)
    out_path = output_dir / "actor_group" / "rank_0.log"
    assert out_path.read_text(encoding="utf-8") == "line-1\nline-2\n"

    with worker_log.open("a", encoding="utf-8") as fp:
        fp.write("line-3\n")
    collector._process_once(logs_dir, new_file_offset_from_start=False)
    assert out_path.read_text(encoding="utf-8") == "line-1\nline-2\nline-3\n"

    collector.stop()


def test_process_once_skips_historical_content_for_new_file_in_loop(tmp_path: Path):
    logs_dir = tmp_path / "ray_logs"
    output_dir = tmp_path / "split_logs"
    logs_dir.mkdir(parents=True)

    worker_log = logs_dir / "worker-w2-j2-456.err"
    worker_log.write_text("old-line\n", encoding="utf-8")

    collector = _new_collector(logs_dir=logs_dir, output_dir=output_dir)
    collector._log_file_map[worker_log] = ("collector_group:1", "1")

    # Background loop behavior: new files start at EOF (historical lines are skipped).
    collector._process_once(logs_dir, new_file_offset_from_start=False)
    out_path = output_dir / "collector_group" / "rank_1.log"
    assert not out_path.exists()

    with worker_log.open("a", encoding="utf-8") as fp:
        fp.write("new-line\n")
    collector._process_once(logs_dir, new_file_offset_from_start=False)
    assert out_path.read_text(encoding="utf-8") == "new-line\n"

    collector.stop()


def test_start_stop_drains_remaining_logs_from_beginning(tmp_path: Path):
    logs_dir = tmp_path / "ray_logs"
    output_dir = tmp_path / "split_logs"
    logs_dir.mkdir(parents=True)

    collector = LocalRayLogCollector(
        logger=logging.getLogger("test.distributed_log_collector.stop"),
        output_dir=output_dir,
        logs_dir=logs_dir,
        poll_interval_s=2.0,
    )
    assert collector.start() is True

    # Let the thread run one iteration with an empty mapping, then wait.
    time.sleep(0.2)

    worker_log = logs_dir / "worker-w3-j3-789.out"
    worker_log.write_text("late-line-1\nlate-line-2\n", encoding="utf-8")
    collector._log_file_map[worker_log] = ("late_group:0", "0")

    # stop() performs drain with new_file_offset_from_start=True, so it should
    # capture complete content even if the background thread did not process it yet.
    collector.stop()

    out_path = output_dir / "late_group" / "rank_0.log"
    assert out_path.read_text(encoding="utf-8") == "late-line-1\nlate-line-2\n"


class CollectorIntegrationWorker(Worker):
    """Worker used for integration testing with Cluster and Ray."""

    def __init__(self):
        super().__init__()

    def emit_test_log(self, token: str) -> int:
        self.log_info(f"collector-integration-token {token}")
        print(f"collector-integration-stdout {token}", flush=True)
        print(f"collector-integration-stderr {token}", file=sys.stderr, flush=True)
        return os.getpid()


def _reset_cluster_singleton() -> None:
    if ray.is_initialized():
        ray.shutdown()
    if hasattr(Cluster, "_instance"):
        instance = getattr(Cluster, "_instance")
        if instance is not None:
            instance._has_initialized = False
        delattr(Cluster, "_instance")
    Cluster.NAMESPACE = Cluster.SYS_NAME


def test_cluster_launch_collects_real_worker_logs(tmp_path: Path):
    out_dir = tmp_path / "cluster_logs"
    token = f"collector-token-{uuid.uuid4().hex}"
    _reset_cluster_singleton()
    tests_root = Path(__file__).resolve().parent
    python_path_entries = [str(tests_root)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        python_path_entries.append(existing_pythonpath)
    python_path_value = os.pathsep.join(python_path_entries)
    cluster_cfg = OmegaConf.create(
        {
            "num_nodes": 1,
            "component_placement": {},
            "node_groups": [
                {
                    "label": "train",
                    "node_ranks": "0",
                    "env_configs": [
                        {
                            "node_ranks": "0",
                            "python_interpreter_path": sys.executable,
                            "env_vars": [{"PYTHONPATH": python_path_value}],
                        }
                    ],
                }
            ],
        }
    )

    cluster = None
    worker_group = None
    try:
        cluster = Cluster(cluster_cfg=cluster_cfg, distributed_log_dir=str(out_dir))
        worker_group = CollectorIntegrationWorker.create_group().launch(
            cluster=cluster,
            placement_strategy=NodePlacementStrategy([0], node_group_label="train"),
            name="collector_integration_group",
        )
        pids = worker_group.emit_test_log(token).wait()
        assert len(pids) == 1
        worker_pid = pids[0]

        collector = cluster._distributed_log_collector
        assert collector is not None
        logs_dir = collector._get_ray_logs_dir()
        assert logs_dir is not None

        collector._stop_event.set()
        if collector._thread is not None:
            collector._thread.join(timeout=10)

        candidates = []
        bind_deadline = time.time() + 30
        while time.time() < bind_deadline and len(candidates) == 0:
            candidates = (
                list(logs_dir.glob(f"worker-*-*-{worker_pid}.out"))
                + list(logs_dir.glob(f"worker-*-{worker_pid}.out"))
                + list(logs_dir.glob(f"worker-*-*-{worker_pid}.err"))
                + list(logs_dir.glob(f"worker-*-{worker_pid}.err"))
            )
            if len(candidates) == 0:
                time.sleep(0.5)
        assert len(candidates) > 0, (
            "Did not find Ray worker log files for launched worker pid "
            f"{worker_pid} under {logs_dir}."
        )
        for candidate in candidates:
            collector._log_file_map[candidate] = ("collector_integration_group:0", "0")
            # Drop any offset the background loop may have recorded so the
            # from-start read below begins at byte 0 and captures the token.
            collector._file_offsets.pop(candidate, None)

        target_log = out_dir / "collector_integration_group" / "rank_0.log"
        deadline = time.time() + 30
        content = ""
        while time.time() < deadline:
            collector._process_once(logs_dir, new_file_offset_from_start=True)
            if target_log.exists():
                content = target_log.read_text(encoding="utf-8")
                if token in content:
                    break
            time.sleep(0.5)
        assert token in content, (
            "Did not find emitted integration token in collected worker log "
            f"within timeout. Current content:\n{content}"
        )
        collector.stop()
    finally:
        if worker_group is not None:
            worker_group._close()
        _reset_cluster_singleton()


class FakeProxy:
    """Stand-in for the tracer ManagerProxy that records events in memory."""

    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class TestTracerManager:
    """Server-side test: the Tracer manager writes events to a JSONL file."""

    def test_record_and_finalize(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "trace.jsonl")
            # Construct the manager as a plain object (no Ray actor needed).
            tracer = Tracer(path)
            tracer.record({"name": "a", "ph": "B", "ts": 1, "pid": "p", "tid": "main"})
            tracer.record({"name": "a", "ph": "E", "ts": 2, "pid": "p", "tid": "main"})
            assert tracer.finalize() == os.path.abspath(path)

            with open(path) as f:
                events = [json.loads(line) for line in f]
            assert [e["ph"] for e in events] == ["B", "E"]
            assert all(e["name"] == "a" for e in events)


class TestTracerEmit:
    """Client-side test: the emit API forwards well-formed events, or no-ops."""

    def teardown_method(self):
        Tracer._unavailable = False
        Tracer._labeled = False

    def test_emit_forwards_events(self, monkeypatch):
        proxy = FakeProxy()
        monkeypatch.setattr(Tracer, "_get", classmethod(lambda cls: proxy))
        monkeypatch.setattr(Tracer, "_pid", staticmethod(lambda: "driver"))
        Tracer._labeled = False

        with Tracer.trace_span("step", cat="runner", args={"i": 0}):
            Tracer.trace_begin("inner", cat="actor")
            Tracer.trace_end("inner", cat="actor")

        @Tracer.trace_func(cat="dec")
        def fn():
            return 7

        assert fn() == 7

        # A one-time process_name metadata event labels the process.
        meta = [e for e in proxy.events if e["ph"] == "M"]
        assert len(meta) == 1
        assert meta[0]["args"]["name"] == "driver"

        # Duration events are well-formed and balanced per name.
        spans = [e for e in proxy.events if e["ph"] in ("B", "E")]
        assert all({"name", "cat", "ph", "ts", "pid", "tid"} <= e.keys() for e in spans)
        step_begin = next(e for e in spans if e["name"] == "step" and e["ph"] == "B")
        assert step_begin["cat"] == "runner" and step_begin["args"] == {"i": 0}
        for name in ("step", "inner", "fn"):
            phs = sorted(e["ph"] for e in spans if e["name"] == name)
            assert phs == ["B", "E"]

    def test_disabled_is_noop(self):
        # When the tracer manager is unavailable, every emit API is a no-op.
        Tracer._unavailable = True

        Tracer.trace_begin("noop")
        Tracer.trace_end("noop")
        with Tracer.trace_span("noop"):
            pass

        @Tracer.trace_func
        def fn():
            return 42

        assert fn() == 42
        assert Tracer._get() is None


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])


@pytest.fixture(autouse=True)
def _reset_profiling_flag():
    """Reset module-level flag between tests for isolation."""
    nv_module._nv_profiling_active = False
    yield
    nv_module._nv_profiling_active = False


class TestStartStop:
    """NvidiaGPUManager.start_profiling / stop_profiling drive torch.cuda.profiler."""

    def test_start_sets_active_and_calls_cuda_profiler_start(self):
        with patch("torch.cuda.profiler.start") as mock_start:
            NvidiaGPUManager.start_profiling(step_idx=5)
        assert NvidiaGPUManager.is_profiling_active() is True
        mock_start.assert_called_once_with()

    def test_stop_clears_active_and_calls_cuda_profiler_stop(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()
        with patch("torch.cuda.profiler.stop") as mock_stop:
            NvidiaGPUManager.stop_profiling()
        assert NvidiaGPUManager.is_profiling_active() is False
        mock_stop.assert_called_once_with()

    def test_double_start_is_idempotent(self):
        with patch("torch.cuda.profiler.start") as mock_start:
            NvidiaGPUManager.start_profiling()
            NvidiaGPUManager.start_profiling()
        assert mock_start.call_count == 1

    def test_stop_without_start_is_noop(self):
        with patch("torch.cuda.profiler.stop") as mock_stop:
            NvidiaGPUManager.stop_profiling()
        mock_stop.assert_not_called()


class TestProfilingRangeSync:
    """NvidiaGPUManager.profiling_range on sync code is transparent off / wraps on."""

    def test_passes_through_when_inactive(self):
        results = []
        with NvidiaGPUManager.profiling_range("test/op"):
            results.append(42)
        assert results == [42]
        assert NvidiaGPUManager.is_profiling_active() is False

    def test_emits_range_when_active(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        with (
            patch("torch.cuda.nvtx.range_push") as mock_range_push,
            patch("torch.cuda.nvtx.range_pop") as mock_range_pop,
        ):
            with NvidiaGPUManager.profiling_range("test/op", color="green"):
                pass

        mock_range_push.assert_called_once_with("test/op")
        mock_range_pop.assert_called_once_with()

    def test_ends_range_on_exception(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        with (
            patch("torch.cuda.nvtx.range_push") as mock_range_push,
            patch("torch.cuda.nvtx.range_pop") as mock_range_pop,
        ):
            with pytest.raises(ValueError, match="oops"):
                with NvidiaGPUManager.profiling_range("test/op"):
                    raise ValueError("oops")

        mock_range_push.assert_called_once_with("test/op")
        mock_range_pop.assert_called_once_with()


class TestProfilingRangeAsync:
    """profiling_range works correctly inside async coroutines."""

    def test_passes_through_when_inactive(self):
        async def coro():
            with NvidiaGPUManager.profiling_range("test/async_op"):
                return 99

        assert asyncio.run(coro()) == 99

    def test_emits_range_when_active(self):
        with patch("torch.cuda.profiler.start"):
            NvidiaGPUManager.start_profiling()

        async def coro():
            with NvidiaGPUManager.profiling_range("test/async_op"):
                return 7

        with (
            patch("torch.cuda.nvtx.range_push") as mock_range_push,
            patch("torch.cuda.nvtx.range_pop") as mock_range_pop,
        ):
            result = asyncio.run(coro())

        assert result == 7
        mock_range_push.assert_called_once_with("test/async_op")
        mock_range_pop.assert_called_once_with()


def test_get_torch_platform_adds_ipc_collect_when_missing(monkeypatch):
    torch_module = ModuleType("torch")
    xpu_platform = SimpleNamespace()
    torch_module.xpu = xpu_platform
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    result = IntelGPUManager.get_torch_platform()

    assert result is xpu_platform
    assert hasattr(xpu_platform, "ipc_collect")
    assert callable(xpu_platform.ipc_collect)
    assert xpu_platform.ipc_collect() is None


def test_get_torch_platform_keeps_existing_ipc_collect(monkeypatch):
    torch_module = ModuleType("torch")
    sentinel = object()

    def _existing_ipc_collect():
        return sentinel

    xpu_platform = SimpleNamespace(ipc_collect=_existing_ipc_collect)
    torch_module.xpu = xpu_platform
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    result = IntelGPUManager.get_torch_platform()

    assert result is xpu_platform
    assert xpu_platform.ipc_collect is _existing_ipc_collect
    assert xpu_platform.ipc_collect() is sentinel


_PATCHER_UNDER_TEST = "_rlinf_utils_patcher_under_test"
_TEST_MODULE_NAMES = (
    "flash_attn",
    "rlinf_test_missing_cuda_dep",
    "rlinf_test_missing_cuda_pkg",
    "rlinf_test_loaded_dep",
)


def _load_patcher_module():
    patcher_path = (
        Path(__file__).resolve().parents[2] / "rlinf" / "utils" / "patcher.py"
    )
    spec = importlib.util.spec_from_file_location(
        _PATCHER_UNDER_TEST,
        patcher_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Patcher = _load_patcher_module().Patcher


def _remove_test_modules() -> None:
    for root_name in _TEST_MODULE_NAMES:
        for module_name in list(sys.modules):
            if module_name == root_name or module_name.startswith(f"{root_name}."):
                del sys.modules[module_name]


@pytest.fixture(autouse=True)
def clean_patcher_state():
    originals = {
        name: sys.modules[name] for name in _TEST_MODULE_NAMES if name in sys.modules
    }
    _remove_test_modules()
    Patcher.clear()
    yield
    Patcher.clear()
    _remove_test_modules()
    sys.modules.update(originals)


def test_skip_import_registers_stub_immediately():
    result = Patcher.skip_import("rlinf_test_missing_cuda_dep")

    assert result is Patcher
    stub_module = sys.modules["rlinf_test_missing_cuda_dep"]
    assert stub_module.__name__ == "rlinf_test_missing_cuda_dep"
    assert stub_module.some_kernel() is None


def test_skip_import_supports_submodule_imports():
    Patcher.skip_import("rlinf_test_missing_cuda_pkg")

    submodule = importlib.import_module("rlinf_test_missing_cuda_pkg.bert_padding")
    namespace = {}
    exec(
        "from rlinf_test_missing_cuda_pkg.bert_padding import unpad_input",
        namespace,
    )

    assert submodule is sys.modules["rlinf_test_missing_cuda_pkg.bert_padding"]
    assert namespace["unpad_input"]() is None


def test_skip_import_preserves_loaded_modules():
    loaded_module = types.ModuleType("rlinf_test_loaded_dep")
    loaded_module.existing_value = object()
    sys.modules["rlinf_test_loaded_dep"] = loaded_module

    Patcher.skip_import("rlinf_test_loaded_dep")

    assert sys.modules["rlinf_test_loaded_dep"] is loaded_module


def test_clear_stub_import_removes_stub_tree_only():
    Patcher.skip_import("rlinf_test_missing_cuda_dep")
    importlib.import_module("rlinf_test_missing_cuda_dep.bert_padding")

    result = Patcher.clear_stub_import("rlinf_test_missing_cuda_dep")

    assert result is Patcher
    assert "rlinf_test_missing_cuda_dep" not in sys.modules
    assert "rlinf_test_missing_cuda_dep.bert_padding" not in sys.modules


def test_clear_stub_import_preserves_real_modules():
    real_module = types.ModuleType("rlinf_test_loaded_dep")
    real_module.__file__ = "/site-packages/rlinf_test_loaded_dep/__init__.py"
    sys.modules["rlinf_test_loaded_dep"] = real_module

    Patcher.clear_stub_import("rlinf_test_loaded_dep")

    assert sys.modules["rlinf_test_loaded_dep"] is real_module


def test_clear_stub_import_removes_flash_attn_stub_tree_only():
    Patcher.skip_import("flash_attn")
    importlib.import_module("flash_attn.bert_padding")

    Patcher.clear_stub_import("flash_attn")

    assert "flash_attn" not in sys.modules
    assert "flash_attn.bert_padding" not in sys.modules


def test_clear_stub_import_preserves_real_flash_attn_module():
    real_flash_attn = types.ModuleType("flash_attn")
    real_flash_attn.__file__ = "/site-packages/flash_attn/__init__.py"
    sys.modules["flash_attn"] = real_flash_attn

    Patcher.clear_stub_import("flash_attn")

    assert sys.modules["flash_attn"] is real_flash_attn


@pytest.mark.parametrize("phase", ["train", "eval"])
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize(
    ("replay_mode", "stop_on_done"),
    [("openpi_chunk", False), ("flow_atomic", False), ("flow_atomic", True), (None, False)],
)
def test_env_worker_chunk_step_contract(phase, terminal, replay_mode, stop_on_done):
    """Train/eval preserve each environment's chunk API and terminal outputs."""
    import inspect

    import torch

    from rlinf.workers.env.env_worker import EnvWorker

    actions = torch.zeros(2, 3, 7)
    obs = {"states": torch.ones(2, 7)}
    final_obs = {"states": torch.full((2, 7), 2.0)}
    rewards = torch.ones(2, 3)
    terminations = torch.zeros(2, 3, dtype=torch.bool)
    terminations[0, -1] = terminal
    truncations = torch.zeros_like(terminations)
    success = torch.tensor([terminal, False])
    calls = []

    def result(chunk_actions):
        torch.testing.assert_close(torch.as_tensor(chunk_actions), actions)
        info = {"episode": {"success_once": success}}
        if terminal:
            info.update(
                final_observation=final_obs,
                final_info={"episode": {"success_once": success}},
            )
        if replay_mode == "flow_atomic":
            info["_raw_chunk_terminations"] = terminations.clone()
            info["_raw_chunk_truncations"] = truncations.clone()
        return [obs], rewards, terminations, truncations, [info]

    class LiberoChunkEnv:
        def chunk_step(self, chunk_actions):
            calls.append(None)
            return result(chunk_actions)

    class ManiSkillChunkEnv:
        def chunk_step(self, chunk_actions, *, stop_on_done):
            calls.append(stop_on_done)
            return result(chunk_actions)

    is_flow = replay_mode == "flow_atomic"
    env = ManiSkillChunkEnv() if is_flow else LiberoChunkEnv()
    env_cfg = {
        "env_type": "maniskill" if is_flow else "libero",
        "auto_reset": True,
        "ignore_terminations": False,
        "stop_on_done": stop_on_done,
    }
    worker = SimpleNamespace(
        cfg=OmegaConf.create({"env": {"train": env_cfg, "eval": env_cfg}}),
        model_cfg=OmegaConf.create(
            {
                "model_type": "flow_policy" if is_flow else "openpi_rlinf",
                "num_action_chunks": 3,
                "action_dim": 7,
                "policy_setup": "panda",
            }
        ),
        env_list=[env],
        eval_env_list=[env],
        _is_ogpo=replay_mode is not None,
        _is_openpi_chunk_ogpo=replay_mode == "openpi_chunk",
        use_external_reward_model=False,
        eval_prev_done=[torch.zeros(2, dtype=torch.bool)],
    )
    if phase == "train":
        output, metrics, payload = inspect.unwrap(EnvWorker.env_interact_step)(
            worker, actions, 0
        )
        torch.testing.assert_close(output.rewards, rewards)
        torch.testing.assert_close(output.dones, terminations)
        if replay_mode is not None:
            torch.testing.assert_close(payload["raw_terminations"], terminations)
            torch.testing.assert_close(payload["raw_truncations"], truncations)
            torch.testing.assert_close(payload["success"], success)
    else:
        output, metrics = inspect.unwrap(EnvWorker.env_evaluate_step)(
            worker, actions, 0
        )
    expected_stop = (True if phase == "eval" else stop_on_done) if is_flow else None
    assert calls == [expected_stop]
    torch.testing.assert_close(output.obs["states"], obs["states"])
    if terminal:
        torch.testing.assert_close(output.final_obs["states"], final_obs["states"])
        torch.testing.assert_close(metrics["success_once"], success[:1])
    else:
        assert output.final_obs is None
        assert metrics == {}


@pytest.mark.parametrize("horizon", [5, 20])
@pytest.mark.parametrize(
    ("boundary", "boundary_index", "bootstrap"),
    [
        ("none", None, True),
        ("termination", -1, False),
        ("truncation", -1, True),
        ("both", -1, False),
        ("termination", 1, False),
        ("truncation", 1, True),
    ],
)
def test_openpi_ogpo_critic_environment_step_targets(
    horizon, boundary, boundary_index, bootstrap
):
    """Discount per environment step and bootstrap through time limits only."""
    import inspect

    import torch

    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.workers.actor.fsdp_ogpo_policy_worker import EmbodiedOGPOFSDPPolicy

    class ConstantQPolicy(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor(values))

        def forward(self, *, forward_type, obs, actions=None, num_samples=None):
            if forward_type == ForwardType.OGPO_SAMPLE:
                return torch.zeros(1, 1, horizon * 7), None, None, None
            assert forward_type == ForwardType.SAC_Q
            return self.values.unsqueeze(0)

    worker = object.__new__(EmbodiedOGPOFSDPPolicy)
    worker.is_openpi_chunk_ogpo = True
    worker.cfg = OmegaConf.create({"algorithm": {"gamma": 0.9}})
    worker.ogpo_cfg = OmegaConf.create({"td_q_aggregation": "mean"})
    worker.q_sample_generator = torch.Generator().manual_seed(0)
    worker.model = ConstantQPolicy([0.25, 0.75])
    worker.target_model = ConstantQPolicy([2.0, 4.0])
    rewards = torch.arange(1, horizon + 1, dtype=torch.float32).unsqueeze(0)
    terminations = torch.zeros(1, horizon, dtype=torch.bool)
    truncations = torch.zeros_like(terminations)
    if boundary in ("termination", "both"):
        terminations[0, boundary_index] = True
    if boundary in ("truncation", "both"):
        truncations[0, boundary_index] = True
    batch = {
        "curr_obs": {"states": torch.zeros(1, 7)},
        "next_obs": {"states": torch.ones(1, 7)},
        "actions": torch.zeros(1, horizon, 7),
        "rewards": rewards,
        "terminations": terminations,
        "dones": terminations | truncations,
        "valid": torch.ones(1, dtype=torch.bool),
    }
    loss, metrics = inspect.unwrap(EmbodiedOGPOFSDPPolicy.forward_critic)(worker, batch)
    length = horizon if boundary_index in (None, -1) else boundary_index + 1
    discounted_reward = sum((i + 1) * 0.9**i for i in range(length))
    expected_target = discounted_reward + (0.9**length * 3.0 if bootstrap else 0)
    expected_loss = ((0.25 - expected_target) ** 2 + (0.75 - expected_target) ** 2) / 2
    torch.testing.assert_close(
        metrics["critic/target_q_mean"], torch.tensor(expected_target)
    )
    torch.testing.assert_close(loss, torch.tensor(expected_loss))
    loss.backward()
    torch.testing.assert_close(
        worker.model.values.grad, worker.model.values.detach() - expected_target
    )
    assert worker.target_model.values.grad is None


@pytest.mark.parametrize(
    ("global_step", "update_step", "warmup", "critic_only", "epochs"),
    [
        (0, 160, 20, False, 8),
        (19, 152, 20, False, 8),
        (20, 0, 20, False, 8),
        (21, 0, 20, False, 1),
        (0, 0, 0, False, 1),
        (20, 160, 20, True, 8),
        (99, 999, 100, False, 8),
        (100, 0, 100, False, 8),
    ],
)
def test_ogpo_warmup_uses_runner_global_step(
    global_step, update_step, warmup, critic_only, epochs
):
    """Actor updates follow runner steps regardless of internal update count."""
    import copy
    import inspect

    import torch

    from rlinf.workers.actor.fsdp_ogpo_policy_worker import EmbodiedOGPOFSDPPolicy

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Parameter(torch.ones(()))
            self.q_head = torch.nn.Parameter(torch.ones(()))

        def clip_grad_norm_(self, max_norm):
            return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    class WarmupWorker(EmbodiedOGPOFSDPPolicy):
        def __init__(self):
            self.model = TinyPolicy()
            self.target_model = copy.deepcopy(self.model).requires_grad_(False)
            self.critic_only = critic_only
            self.update_step = update_step
            self.ogpo_cfg = OmegaConf.create(
                {"q_warmup_steps": warmup, "q_utd": 4, "pi_utd": 1}
            )
            self.cfg = OmegaConf.create(
                {
                    "actor": {
                        "optim": {"clip_grad": 100},
                        "critic_optim": {"clip_grad": 100},
                    }
                }
            )
            self.optimizer = torch.optim.SGD([self.model.actor], lr=0.1)
            self.qf_optimizer = torch.optim.SGD([self.model.q_head], lr=0.1)
            self.lr_scheduler = None
            self.qf_lr_scheduler = None

        def _sample_train_micro_batches(self):
            return [{"actions": torch.zeros(1, 1)}]

        def forward_critic(self, batch):
            loss = self.model.q_head.square()
            return loss, {"critic/loss": loss.detach()}

        def forward_actor(self, batch):
            loss = self.model.actor.square()
            return loss, {"actor/loss": loss.detach()}

    worker = WarmupWorker()
    inspect.unwrap(EmbodiedOGPOFSDPPolicy.set_global_step)(worker, global_step)
    for _ in range(epochs):
        metrics = inspect.unwrap(EmbodiedOGPOFSDPPolicy.update_one_epoch)(worker)
        should_update = not critic_only and global_step >= warmup
        assert ("actor/grad_norm" in metrics) == should_update
        worker.update_step += 1
    expected_actor = 0.8**epochs if should_update else 1.0
    torch.testing.assert_close(worker.model.actor, torch.tensor(expected_actor))
    torch.testing.assert_close(worker.model.q_head, torch.tensor(0.8 ** (4 * epochs)))
