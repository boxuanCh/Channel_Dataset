"""Canonical public tensor-axis names."""

from enum import StrEnum


class Axis(StrEnum):
    """Stable semantic names used by public array containers."""

    BATCH = "batch"
    RX = "rx"
    RX_ANTENNA = "rx_antenna"
    TX = "tx"
    TX_ANTENNA = "tx_antenna"
    PATH = "path"
    CLUSTER = "cluster"
    RAY = "ray"
    TIME = "time"
    OUTPUT_TIME = "output_time"
    LAG = "lag"
    SYMBOL = "symbol"
    SUBCARRIER = "subcarrier"
    COORDINATE = "coordinate"
    ORIENTATION = "orientation"
    STRONGEST_CLUSTER = "strongest_cluster"


CIR_COEFFICIENT_AXES = (
    Axis.BATCH,
    Axis.RX,
    Axis.RX_ANTENNA,
    Axis.TX,
    Axis.TX_ANTENNA,
    Axis.PATH,
    Axis.TIME,
)
CIR_DELAY_AXES = (Axis.BATCH, Axis.RX, Axis.TX, Axis.PATH)
OFDM_RESPONSE_AXES = (
    Axis.BATCH,
    Axis.RX,
    Axis.RX_ANTENNA,
    Axis.TX,
    Axis.TX_ANTENNA,
    Axis.SYMBOL,
    Axis.SUBCARRIER,
)
TIME_RESPONSE_AXES = (
    Axis.BATCH,
    Axis.RX,
    Axis.RX_ANTENNA,
    Axis.TX,
    Axis.TX_ANTENNA,
    Axis.OUTPUT_TIME,
    Axis.LAG,
)
FLAT_RESPONSE_AXES = (
    Axis.BATCH,
    Axis.RX,
    Axis.RX_ANTENNA,
    Axis.TX,
    Axis.TX_ANTENNA,
)
LINK_AXES = (Axis.BATCH, Axis.RX, Axis.TX)
CLUSTER_AXES = (*LINK_AXES, Axis.CLUSTER)
RAY_AXES = (*CLUSTER_AXES, Axis.RAY)
