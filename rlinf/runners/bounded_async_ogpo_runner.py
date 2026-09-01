# Copyright 2026 The RLinf Authors.
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

"""Bounded one-batch asynchronous pipeline for embodied OGPO."""

import time

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import WorkerGroupFuncResult as Handle


class BoundedAsyncOGPOEmbodiedRunner(EmbodiedRunner):
    """Overlap training on replay data with collection of one next batch."""

    def _collect_rollout(self) -> None:
        env_handle: Handle = self.env.interact(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
            reward_channel=self.reward_channel,
            actor_channel=self.actor_channel,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        self.actor.recv_rollout_trajectories(
            input_channel=self.actor_channel
        ).wait()
        rollout_handle.wait()
        env_handle.wait()
        self.actor.compute_advantages_and_returns().wait()

    def run(self) -> None:
        if self.reward is not None:
            raise ValueError("Bounded async OGPO does not support a reward worker")
        if self.global_step >= self.max_steps:
            self._finish_run()
            return

        start_step = self.global_step
        start_time = time.time()

        # Prime the pipeline with exactly one replay batch.
        self.actor.set_global_step(self.global_step).wait()
        self.rollout.set_global_step(self.global_step).wait()
        self.update_rollout_weights()
        self._collect_rollout()

        while self.global_step < self.max_steps:
            step = self.global_step
            self.actor.set_global_step(step).wait()
            self.rollout.set_global_step(step).wait()

            # Drain the final pending batch without collecting a surplus batch.
            if step + 1 >= self.max_steps:
                self.actor.run_training().wait()
                self.global_step += 1
                self._maybe_eval_and_checkpoint(step)
                break

            with self.timer("step", trace_args={"step_idx": step}):
                with self.timer("sync_weights"):
                    if step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()

                env_handle: Handle = self.env.interact(
                    input_channel=self.env_channel,
                    rollout_channel=self.rollout_channel,
                    reward_channel=self.reward_channel,
                    actor_channel=self.actor_channel,
                )
                rollout_handle: Handle = self.rollout.generate(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                )
                actor_training_handle: Handle = self.actor.run_training()
                recv_handle: Handle = self.actor.recv_rollout_trajectories(
                    input_channel=self.actor_channel
                )

                actor_training_metrics = actor_training_handle.wait()
                recv_handle.wait()
                rollout_handle.wait()
                actor_rollout_metrics = (
                    self.actor.compute_advantages_and_returns().wait()
                )

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(step)

            self._log_step_metrics(
                step=step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                actor_training_handle=actor_training_handle,
                reward_handle=None,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()
