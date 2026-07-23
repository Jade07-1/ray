"""Unit tests for V2-fast upscaling (fast path in reconciler)."""

import math
import os
import sys
import time
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from ray.autoscaler.v2.instance_manager.config import NodeTypeConfig
from ray.autoscaler.v2.instance_manager.reconciler import (
    Reconciler,
    _check_fast_path_degradation,
    compute_fast_target,
    is_homogeneous_demand,
)
from ray.autoscaler.v2.schema import AutoscalerInstance
from ray.core.generated.autoscaler_pb2 import (
    AutoscalingState,
    ClusterResourceState,
    GangResourceRequest,
    ResourceRequest,
    ResourceRequestByCount,
)
from ray.core.generated.instance_manager_pb2 import (
    Instance as IMInstance,
    NodeKind,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_resource_request_by_count(
    bundle: Dict[str, float], count: int
) -> ResourceRequestByCount:
    req = ResourceRequest()
    for k, v in bundle.items():
        req.resources_bundle[k] = v
    return ResourceRequestByCount(request=req, count=count)


def _make_ray_state(
    pending_requests: Optional[List[ResourceRequestByCount]] = None,
    gang_requests: Optional[List[GangResourceRequest]] = None,
) -> ClusterResourceState:
    state = ClusterResourceState()
    if pending_requests:
        for r in pending_requests:
            state.pending_resource_requests.append(r)
    if gang_requests:
        for g in gang_requests:
            state.pending_gang_resource_requests.append(g)
    return state


def _make_autoscaling_config(
    worker_type: str = "worker",
    head_type: str = "head",
    worker_resources: Optional[Dict[str, float]] = None,
    max_workers: Optional[int] = None,
) -> MagicMock:
    if worker_resources is None:
        worker_resources = {"CPU": 8, "memory": 32_000_000_000}

    config = MagicMock()
    config.get_head_node_type.return_value = head_type
    config.get_node_type_configs.return_value = {
        head_type: NodeTypeConfig(
            name=head_type,
            resources={"CPU": 8},
            min_worker_nodes=0,
            max_worker_nodes=1,
            labels={},
            launch_config_hash="h",
        ),
        worker_type: NodeTypeConfig(
            name=worker_type,
            resources=worker_resources,
            min_worker_nodes=0,
            max_worker_nodes=max_workers or 10000,
            labels={},
            launch_config_hash="w",
        ),
    }
    config.get_max_num_worker_nodes.return_value = max_workers
    config.get_max_num_nodes.return_value = (max_workers + 1) if max_workers else None
    return config


def _make_instance(
    instance_type: str = "worker",
    status: int = IMInstance.RAY_RUNNING,
) -> AutoscalerInstance:
    im = IMInstance()
    im.instance_id = f"inst-{id(im)}"
    im.instance_type = instance_type
    im.status = status
    im.status_history.append(
        IMInstance.StatusHistory(instance_status=status, timestamp_ns=time.time_ns())
    )
    return AutoscalerInstance(im_instance=im)


# ---------------------------------------------------------------------------
# Tests for is_homogeneous_demand
# ---------------------------------------------------------------------------


class TestIsHomogeneousDemand:
    def test_triggers_on_large_homogeneous_demand(self):
        """Large batch of identical tasks -> fast path."""
        requests = [_make_resource_request_by_count({"CPU": 1}, 500)]
        state = _make_ray_state(pending_requests=requests)
        assert is_homogeneous_demand(state) is True

    def test_not_triggered_below_threshold(self):
        """Below threshold -> normal path."""
        requests = [_make_resource_request_by_count({"CPU": 1}, 50)]
        state = _make_ray_state(pending_requests=requests)
        assert is_homogeneous_demand(state) is False

    def test_not_triggered_with_gang_requests(self):
        """Presence of Placement Group demands -> normal path."""
        requests = [_make_resource_request_by_count({"CPU": 1}, 500)]
        gang = GangResourceRequest()
        req = gang.requests.add()
        req.resources_bundle["CPU"] = 1
        state = _make_ray_state(pending_requests=requests, gang_requests=[gang])
        assert is_homogeneous_demand(state) is False

    def test_not_triggered_heterogeneous(self):
        """Heterogeneous demands -> normal path."""
        requests = [
            _make_resource_request_by_count({"CPU": 1}, 50),
            _make_resource_request_by_count({"GPU": 1}, 60),
        ]
        state = _make_ray_state(pending_requests=requests)
        assert is_homogeneous_demand(state) is False

    def test_dominance_ratio_boundary_above(self):
        """95% dominance -> fast path."""
        # 950 same + 50 different = 95% dominance
        requests = [
            _make_resource_request_by_count({"CPU": 1}, 950),
            _make_resource_request_by_count({"GPU": 1}, 50),
        ]
        state = _make_ray_state(pending_requests=requests)
        assert is_homogeneous_demand(state) is True

    def test_dominance_ratio_boundary_below(self):
        """94% dominance -> normal path."""
        # 940 same + 60 different = 94% dominance
        requests = [
            _make_resource_request_by_count({"CPU": 1}, 940),
            _make_resource_request_by_count({"GPU": 1}, 60),
        ]
        state = _make_ray_state(pending_requests=requests)
        assert is_homogeneous_demand(state) is False


# ---------------------------------------------------------------------------
# Tests for compute_fast_target
# ---------------------------------------------------------------------------


class TestComputeFastTarget:
    def test_basic_cpu_only(self):
        """1000 tasks * 1 CPU each / 8 CPU per node = 125 nodes."""
        requests = [_make_resource_request_by_count({"CPU": 1}, 1000)]
        state = _make_ray_state(pending_requests=requests)
        config = _make_autoscaling_config(worker_resources={"CPU": 8, "memory": 32e9})
        node_type, count = compute_fast_target(state, config, [])
        assert node_type == "worker"
        assert count == math.ceil(1000 / 8)  # 125

    def test_multi_dimension_takes_max(self):
        """Multi-dimension resource: takes the bottleneck dimension."""
        # 1000 tasks: each needs 1 CPU + 8GB memory
        # Node: 8 CPU + 32 GB memory
        # CPU dim: ceil(1000/8) = 125
        # Mem dim: ceil(8000e9/32e9) = 250 -> bottleneck
        requests = [
            _make_resource_request_by_count({"CPU": 1, "memory": 8_000_000_000}, 1000)
        ]
        state = _make_ray_state(pending_requests=requests)
        config = _make_autoscaling_config(
            worker_resources={"CPU": 8, "memory": 32_000_000_000}
        )
        node_type, count = compute_fast_target(state, config, [])
        assert node_type == "worker"
        assert count == 250

    def test_subtracts_existing_instances(self):
        """Existing in-flight instances are deducted from needed count."""
        requests = [_make_resource_request_by_count({"CPU": 1}, 1000)]
        state = _make_ray_state(pending_requests=requests)
        config = _make_autoscaling_config(worker_resources={"CPU": 8})

        # 50 existing QUEUED instances
        existing = [_make_instance("worker", IMInstance.QUEUED) for _ in range(50)]
        node_type, count = compute_fast_target(state, config, existing)
        # Need 125 - 50 existing = 75
        assert count == 75

    def test_no_duplicate_on_second_reconcile(self):
        """Second reconcile round with existing QUEUED should not re-create."""
        requests = [_make_resource_request_by_count({"CPU": 1}, 1000)]
        state = _make_ray_state(pending_requests=requests)
        config = _make_autoscaling_config(worker_resources={"CPU": 8})

        # All 125 already QUEUED from first round
        existing = [_make_instance("worker", IMInstance.QUEUED) for _ in range(125)]
        node_type, count = compute_fast_target(state, config, existing)
        assert count == 0

    def test_max_workers_constraint(self):
        """max_workers caps the launch count."""
        requests = [_make_resource_request_by_count({"CPU": 1}, 40000)]
        state = _make_ray_state(pending_requests=requests)
        config = _make_autoscaling_config(worker_resources={"CPU": 8}, max_workers=4000)
        node_type, count = compute_fast_target(state, config, [])
        # Would need 5000 but max=4000
        assert count == 4000

    def test_excludes_stop_requested(self):
        """RAY_STOP_REQUESTED instances are not counted as existing capacity."""
        requests = [_make_resource_request_by_count({"CPU": 1}, 80)]
        state = _make_ray_state(pending_requests=requests)
        config = _make_autoscaling_config(worker_resources={"CPU": 8})

        # 10 nodes exist but 5 are being drained
        existing = [
            _make_instance("worker", IMInstance.RAY_RUNNING) for _ in range(5)
        ] + [_make_instance("worker", IMInstance.RAY_STOP_REQUESTED) for _ in range(5)]
        node_type, count = compute_fast_target(state, config, existing)
        # Need ceil(80/8)=10, existing active=5, so launch 5
        assert count == 5


# ---------------------------------------------------------------------------
# Tests for degradation check
# ---------------------------------------------------------------------------


class TestFastPathDegradation:
    def test_no_degradation_without_failures(self):
        instances = [_make_instance("worker", IMInstance.REQUESTED) for _ in range(10)]
        assert _check_fast_path_degradation(instances) is False

    def test_degradation_on_high_failure_rate(self):
        """Failure rate > 30% -> degrade."""
        instances = [
            _make_instance("worker", IMInstance.REQUESTED) for _ in range(6)
        ] + [_make_instance("worker", IMInstance.ALLOCATION_FAILED) for _ in range(4)]
        # 4/(6+4) = 40% > 30%
        assert _check_fast_path_degradation(instances) is True

    def test_no_degradation_at_boundary(self):
        """Failure rate exactly 30% -> no degradation (> not >=)."""
        instances = [
            _make_instance("worker", IMInstance.REQUESTED) for _ in range(7)
        ] + [_make_instance("worker", IMInstance.ALLOCATION_FAILED) for _ in range(3)]
        # 3/10 = 30%, boundary -> no degradation
        assert _check_fast_path_degradation(instances) is False

    def test_stale_failures_ignored(self):
        """Failures older than 30s should not trigger degradation."""
        # Create instances with old timestamps (60s ago)
        old_ts = time.time_ns() - 60 * 10**9
        stale_failed = []
        for _ in range(5):
            im = IMInstance()
            im.instance_id = f"inst-{id(im)}"
            im.instance_type = "worker"
            im.status = IMInstance.ALLOCATION_FAILED
            im.status_history.append(
                IMInstance.StatusHistory(
                    instance_status=IMInstance.ALLOCATION_FAILED,
                    timestamp_ns=old_ts,
                )
            )
            stale_failed.append(AutoscalerInstance(im_instance=im))

        # Even though 5/5 = 100% failure, they're all stale -> no degradation
        assert _check_fast_path_degradation(stale_failed) is False


# ---------------------------------------------------------------------------
# Integration test: fast path in _scale_cluster
# ---------------------------------------------------------------------------


class TestScaleClusterFastPath:
    def _make_instance_manager_mock(self, instances: List[IMInstance]):
        im = MagicMock()
        state_mock = MagicMock()
        state_mock.instances = instances
        state_mock.version = 1
        reply_mock = MagicMock()
        reply_mock.status.code = 1  # StatusCode.OK
        reply_mock.state = state_mock
        im.get_instance_manager_state.return_value = reply_mock

        update_reply = MagicMock()
        update_reply.status.code = 1  # StatusCode.OK
        im.update_instance_manager_state.return_value = update_reply
        return im

    def _make_head_instance(self):
        head = IMInstance()
        head.instance_id = "head-1"
        head.instance_type = "head"
        head.status = IMInstance.RAY_RUNNING
        head.node_kind = NodeKind.HEAD
        head.status_history.append(
            IMInstance.StatusHistory(
                instance_status=IMInstance.RAY_RUNNING,
                timestamp_ns=time.time_ns(),
            )
        )
        return head

    @patch(
        "ray.autoscaler.v2.instance_manager.reconciler."
        "AUTOSCALER_FAST_UPSCALING_ENABLED",
        1,
    )
    def test_fast_path_creates_queued_instances(self):
        """Fast path triggers and creates correct number of QUEUED instances."""
        head = self._make_head_instance()
        im = self._make_instance_manager_mock([head])

        requests = [_make_resource_request_by_count({"CPU": 1}, 200)]
        ray_state = _make_ray_state(pending_requests=requests)

        scheduler = MagicMock()
        sched_reply = MagicMock()
        sched_reply.to_launch = []
        sched_reply.to_terminate = []
        sched_reply.to_ippr = []
        sched_reply.infeasible_resource_requests = []
        sched_reply.infeasible_gang_resource_requests = []
        sched_reply.infeasible_cluster_resource_constraints = []
        scheduler.schedule.return_value = sched_reply

        config = _make_autoscaling_config(worker_resources={"CPU": 8})
        config.provider = "kuberay"
        config.get_idle_timeout_s.return_value = None
        config.disable_launch_config_check.return_value = True

        cloud_provider = MagicMock()
        cloud_provider.__class__ = type("FakeProvider", (), {})

        autoscaling_state = AutoscalingState()

        Reconciler._scale_cluster(
            autoscaling_state=autoscaling_state,
            instance_manager=im,
            ray_state=ray_state,
            scheduler=scheduler,
            autoscaling_config=config,
            cloud_provider=cloud_provider,
        )

        # Verify update was called
        assert im.update_instance_manager_state.called
        call_args = im.update_instance_manager_state.call_args
        request = call_args.kwargs.get("request") or call_args[1].get(
            "request", call_args[0][0] if call_args[0] else None
        )
        updates = list(request.updates)
        queued_updates = [
            u for u in updates if u.new_instance_status == IMInstance.QUEUED
        ]
        # ceil(200/8) = 25
        assert len(queued_updates) == 25
        for u in queued_updates:
            assert u.instance_type == "worker"
            assert "fast-path" in u.details

    @patch(
        "ray.autoscaler.v2.instance_manager.reconciler."
        "AUTOSCALER_FAST_UPSCALING_ENABLED",
        0,
    )
    def test_normal_path_when_disabled(self):
        """When fast upscaling is disabled, uses normal scheduler path."""
        head = self._make_head_instance()
        im = self._make_instance_manager_mock([head])

        requests = [_make_resource_request_by_count({"CPU": 1}, 200)]
        ray_state = _make_ray_state(pending_requests=requests)

        scheduler = MagicMock()
        sched_reply = MagicMock()
        sched_reply.to_launch = []
        sched_reply.to_terminate = []
        sched_reply.to_ippr = []
        sched_reply.infeasible_resource_requests = []
        sched_reply.infeasible_gang_resource_requests = []
        sched_reply.infeasible_cluster_resource_constraints = []
        scheduler.schedule.return_value = sched_reply

        config = _make_autoscaling_config(worker_resources={"CPU": 8})
        config.provider = "kuberay"
        config.get_idle_timeout_s.return_value = None
        config.disable_launch_config_check.return_value = True

        cloud_provider = MagicMock()
        cloud_provider.__class__ = type("FakeProvider", (), {})

        autoscaling_state = AutoscalingState()

        Reconciler._scale_cluster(
            autoscaling_state=autoscaling_state,
            instance_manager=im,
            ray_state=ray_state,
            scheduler=scheduler,
            autoscaling_config=config,
            cloud_provider=cloud_provider,
        )

        # With disabled fast path and empty to_launch, no QUEUED created
        scheduler.schedule.assert_called_once()

    @patch(
        "ray.autoscaler.v2.instance_manager.reconciler."
        "AUTOSCALER_FAST_UPSCALING_ENABLED",
        1,
    )
    def test_idle_termination_suppressed_during_fast_path(self):
        """Fast path disables idle termination to prevent scale-up/down conflict."""
        head = self._make_head_instance()
        worker = IMInstance()
        worker.instance_id = "worker-idle-1"
        worker.instance_type = "worker"
        worker.status = IMInstance.RAY_RUNNING
        worker.node_kind = NodeKind.WORKER
        worker.cloud_instance_id = "cloud-1"
        worker.status_history.append(
            IMInstance.StatusHistory(
                instance_status=IMInstance.RAY_RUNNING,
                timestamp_ns=time.time_ns(),
            )
        )

        im = self._make_instance_manager_mock([head, worker])

        requests = [_make_resource_request_by_count({"CPU": 1}, 200)]
        ray_state = _make_ray_state(pending_requests=requests)

        scheduler = MagicMock()
        sched_reply = MagicMock()
        sched_reply.to_launch = []
        sched_reply.to_terminate = []
        sched_reply.to_ippr = []
        sched_reply.infeasible_resource_requests = []
        sched_reply.infeasible_gang_resource_requests = []
        sched_reply.infeasible_cluster_resource_constraints = []
        scheduler.schedule.return_value = sched_reply

        config = _make_autoscaling_config(worker_resources={"CPU": 8})
        config.provider = "kuberay"
        config.get_idle_timeout_s.return_value = 300
        config.disable_launch_config_check.return_value = True

        cloud_provider = MagicMock()
        cloud_provider.__class__ = type("FakeProvider", (), {})

        autoscaling_state = AutoscalingState()

        Reconciler._scale_cluster(
            autoscaling_state=autoscaling_state,
            instance_manager=im,
            ray_state=ray_state,
            scheduler=scheduler,
            autoscaling_config=config,
            cloud_provider=cloud_provider,
        )

        # Verify scheduler was called with idle_timeout_s=None
        # (idle termination suppressed during fast path)
        sched_call_args = scheduler.schedule.call_args[0][0]
        assert sched_call_args.idle_timeout_s is None

    @patch(
        "ray.autoscaler.v2.instance_manager.reconciler."
        "AUTOSCALER_FAST_UPSCALING_ENABLED",
        1,
    )
    def test_degradation_falls_back_to_normal_path(self):
        """High failure rate -> degradation -> uses normal scheduler path."""
        head = self._make_head_instance()

        # Create instances with high failure rate
        failed_instances = []
        for i in range(4):
            inst = IMInstance()
            inst.instance_id = f"failed-{i}"
            inst.instance_type = "worker"
            inst.status = IMInstance.ALLOCATION_FAILED
            inst.status_history.append(
                IMInstance.StatusHistory(
                    instance_status=IMInstance.ALLOCATION_FAILED,
                    timestamp_ns=time.time_ns(),
                )
            )
            failed_instances.append(inst)
        requested_instances = []
        for i in range(6):
            inst = IMInstance()
            inst.instance_id = f"requested-{i}"
            inst.instance_type = "worker"
            inst.status = IMInstance.REQUESTED
            inst.status_history.append(
                IMInstance.StatusHistory(
                    instance_status=IMInstance.REQUESTED,
                    timestamp_ns=time.time_ns(),
                )
            )
            requested_instances.append(inst)

        all_instances = [head] + failed_instances + requested_instances
        im = self._make_instance_manager_mock(all_instances)

        requests = [_make_resource_request_by_count({"CPU": 1}, 200)]
        ray_state = _make_ray_state(pending_requests=requests)

        scheduler = MagicMock()
        sched_reply = MagicMock()
        sched_reply.to_launch = []
        sched_reply.to_terminate = []
        sched_reply.to_ippr = []
        sched_reply.infeasible_resource_requests = []
        sched_reply.infeasible_gang_resource_requests = []
        sched_reply.infeasible_cluster_resource_constraints = []
        scheduler.schedule.return_value = sched_reply

        config = _make_autoscaling_config(worker_resources={"CPU": 8})
        config.provider = "kuberay"
        config.get_idle_timeout_s.return_value = None
        config.disable_launch_config_check.return_value = True

        cloud_provider = MagicMock()
        cloud_provider.__class__ = type("FakeProvider", (), {})

        autoscaling_state = AutoscalingState()

        Reconciler._scale_cluster(
            autoscaling_state=autoscaling_state,
            instance_manager=im,
            ray_state=ray_state,
            scheduler=scheduler,
            autoscaling_config=config,
            cloud_provider=cloud_provider,
        )

        # Should have gone through normal path (scheduler.schedule called)
        # and no fast-path QUEUED instances created (since to_launch is empty)
        scheduler.schedule.assert_called_once()
        if im.update_instance_manager_state.called:
            call_args = im.update_instance_manager_state.call_args
            request = call_args.kwargs.get("request") or call_args[0][0]
            updates = list(request.updates)
            fast_path_updates = [u for u in updates if "fast-path" in (u.details or "")]
            assert len(fast_path_updates) == 0


if __name__ == "__main__":
    if os.environ.get("PARALLEL_CI"):
        sys.exit(pytest.main(["-n", "auto", "--boxed", "-vs", __file__]))
    else:
        sys.exit(pytest.main(["-sv", __file__]))
