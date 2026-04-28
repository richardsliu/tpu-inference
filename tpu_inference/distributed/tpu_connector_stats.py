"""Stats and Prometheus metrics for the NIXL connector."""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine


@dataclass
class TpuKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics"""

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def record_failed_transfer(self):
        """Record a failed TPU KV transfer operation."""
        self.data["num_failed_transfers"].append(1)

    def reset(self):
        # Must be serializable
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "bytes_transferred": [],
            "num_failed_transfers": [],
        }

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        xfer_time = np.asarray(self.data["transfer_duration"])
        # Convert to MB for CLI logging.
        mb = np.asarray(self.data["bytes_transferred"]) / 2**20

        total_mb = mb.sum()
        avg_mb = total_mb / self.num_successful_transfers

        total_time_seconds = xfer_time.sum()
        throughput_mb_s = total_mb / total_time_seconds

        return {
            "Num successful transfers": self.num_successful_transfers,
            "Avg xfer time (ms)": round(xfer_time.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(xfer_time, 90).item() * 1e3, 3),
            "Avg MB per transfer": round(avg_mb, 3),
            "Throughput (MB/s)": round(throughput_mb_s, 3),
        }

    def is_empty(self) -> bool:
        return (
            self.num_successful_transfers == 0
            and len(self.data["num_failed_transfers"]) == 0
        )


    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class TpuKVConnectorPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        buckets = [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            5.0,
        ]
        tpu_histogram_xfer_time = self._histogram_cls(
            name="vllm:tpu_xfer_time_seconds",
            documentation="Histogram of transfer duration for TPU KV Cache transfers.",
            buckets=buckets[1:],
            labelnames=labelnames,
        )
        self.tpu_histogram_xfer_time = create_metric_per_engine(
            tpu_histogram_xfer_time, self.per_engine_labelvalues
        )
        # uniform 2kb to 16gb range
        buckets = [2 ** (10 + i) for i in range(1, 25, 2)]
        tpu_histogram_bytes_transferred = self._histogram_cls(
            name="vllm:tpu_bytes_transferred",
            documentation="Histogram of bytes transferred per TPU KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.tpu_histogram_bytes_transferred = create_metric_per_engine(
            tpu_histogram_bytes_transferred, self.per_engine_labelvalues
        )
        counter_nixl_num_failed_transfers = self._counter_cls(
            name="vllm:tpu_num_failed_transfers",
            documentation="Number of failed TPU KV Cache transfers.",
            labelnames=labelnames,
        )
        self.counter_tpu_num_failed_transfers = create_metric_per_engine(
            counter_tpu_num_failed_transfers, self.per_engine_labelvalues
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for prom_obj, list_item_key in zip(
            [
                self.tpu_histogram_xfer_time,
                self.tpu_histogram_bytes_transferred,
            ],
            [
                "transfer_duration",
                "bytes_transferred",
            ],
        ):
            for list_item in transfer_stats_data[list_item_key]:
                prom_obj[engine_idx].observe(list_item)

        for counter_obj, counter_item_key in zip(
            [
                self.counter_nixl_num_failed_transfers,
            ],
            ["num_failed_transfers"],
        ):
            for list_item in transfer_stats_data[counter_item_key]:
                counter_obj[engine_idx].inc(list_item)
