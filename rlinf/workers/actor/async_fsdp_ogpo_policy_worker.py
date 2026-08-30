# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import queue
import threading
import time

import torch
from omegaconf import DictConfig

from rlinf.data.schema.embodied_types import Trajectory
from rlinf.scheduler import Channel, Worker
from rlinf.workers.actor.fsdp_ogpo_policy_worker import EmbodiedOGPOFSDPPolicy


class AsyncEmbodiedOGPOFSDPPolicy(EmbodiedOGPOFSDPPolicy):
    """Asynchronous OGPO actor with learner-owned replay buffers."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.should_stop = False
        self._recv_error: Exception | None = None
        self._recv_rollout_thread: threading.Thread | None = None
        self._accept_rollout_trajectories = threading.Event()
        self._accept_rollout_trajectories.set()
        self._recv_queue = queue.Queue(
            maxsize=int(self.cfg.actor.get("recv_queue_size", 256))
        )

    async def recv_rollout_trajectories(self, input_channel: Channel) -> bool:
        if self._recv_rollout_thread is not None:
            raise RuntimeError("OGPO rollout receiver is already running")
        self._recv_rollout_thread = threading.Thread(
            target=self._recv_rollout_thread_main,
            args=(input_channel,),
            daemon=True,
        )
        self._recv_rollout_thread.start()
        return True

    def _recv_rollout_thread_main(self, input_channel: Channel) -> None:
        try:
            while not self.should_stop:
                try:
                    message = input_channel.get_nowait()
                except asyncio.QueueEmpty:
                    time.sleep(0.05)
                    continue

                if isinstance(message, Trajectory):
                    episodes = [message]
                elif isinstance(message, list) and all(
                    isinstance(episode, Trajectory) for episode in message
                ):
                    episodes = message
                else:
                    raise TypeError(
                        f"Unsupported async OGPO replay message: {type(message)}"
                    )

                if not self._accept_rollout_trajectories.is_set():
                    continue
                for episode in episodes:
                    while (
                        not self.should_stop
                        and self._accept_rollout_trajectories.is_set()
                    ):
                        try:
                            self._recv_queue.put(episode, timeout=0.1)
                            break
                        except queue.Full:
                            continue
        except Exception as exc:
            self._recv_error = exc

    def _raise_receiver_error(self) -> None:
        if self._recv_error is not None:
            raise RuntimeError(
                "Async OGPO rollout receiver failed"
            ) from self._recv_error

    def _drain_received_trajectories(self, max_trajectories: int) -> int:
        episodes: list[Trajectory] = []
        while len(episodes) < max_trajectories:
            try:
                episodes.append(self._recv_queue.get_nowait())
            except queue.Empty:
                break
        if episodes and self._accept_rollout_trajectories.is_set():
            self._ingest_episodes(episodes)
        return len(episodes)

    async def freeze_replay_buffer(self) -> None:
        """Discard future online trajectories while continuing channel drains."""
        self._accept_rollout_trajectories.clear()
        while True:
            try:
                self._recv_queue.get_nowait()
            except queue.Empty:
                break

    async def _wait_for_replay_buffer_ready(self, min_buffer_size: int) -> bool:
        drain_limit = int(
            self.cfg.actor.get("recv_drain_max_trajectories", 256)
        )
        while not self.should_stop:
            self._raise_receiver_error()
            self._drain_received_trajectories(drain_limit)
            if self._replay_buffer_ready(min_buffer_size):
                return True
            await asyncio.sleep(0.1)
        return False

    @Worker.timer("run_training")
    async def run_training(self) -> dict:
        min_buffer_size = int(
            self.cfg.algorithm.replay_buffer.get("min_buffer_size", 1)
        )
        if not await self._wait_for_replay_buffer_ready(min_buffer_size):
            return {}

        self.model.train()
        metrics: dict = {}
        drain_limit = int(
            self.cfg.actor.get("recv_drain_max_trajectories", 256)
        )
        for _ in range(int(self.cfg.algorithm.get("update_epoch", 1))):
            self._raise_receiver_error()
            self._drain_received_trajectories(drain_limit)
            epoch_metrics = self.update_one_epoch()
            for key, values in epoch_metrics.items():
                if isinstance(values, list):
                    metrics.setdefault(key, []).extend(values)
                else:
                    metrics.setdefault(key, []).append(values)
            self.update_step += 1
            await asyncio.sleep(0)

        result = self.process_train_metrics(metrics)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return result

    async def stop(self) -> None:
        self.should_stop = True
        recv_thread = self._recv_rollout_thread
        if recv_thread is not None and recv_thread.is_alive():
            await asyncio.to_thread(recv_thread.join, 5)
        if recv_thread is not None and recv_thread.is_alive():
            raise RuntimeError("Async OGPO rollout receiver did not stop")
        self._raise_receiver_error()
