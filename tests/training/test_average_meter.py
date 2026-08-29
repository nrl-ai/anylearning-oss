"""The running average that reports training loss to the UI.

One non-finite value used to poison it permanently: `sum` is a running total,
so a single inf made every later average inf as well, and the chart read "inf"
for the rest of a run that had recovered on the next iteration. float16
gradient overflow produces exactly that value.
"""

import math

from anylearning.training.models.utils import AverageMeter


def test_it_averages():
    meter = AverageMeter()
    meter.update(2.0)
    meter.update(4.0)
    assert meter.avg == 3.0


def test_a_weighted_update_counts_n_times():
    meter = AverageMeter()
    meter.update(1.0, n=3)
    meter.update(5.0, n=1)
    assert meter.avg == 2.0


def test_one_infinity_does_not_poison_every_later_average():
    meter = AverageMeter()
    meter.update(2.0)
    meter.update(float("inf"))
    meter.update(4.0)
    assert math.isfinite(meter.avg)
    assert meter.avg == 3.0


def test_nan_is_dropped_too():
    meter = AverageMeter()
    meter.update(2.0)
    meter.update(float("nan"))
    assert meter.avg == 2.0


def test_dropped_values_are_counted_rather_than_hidden():
    meter = AverageMeter()
    meter.update(float("inf"), n=2)
    assert meter.skipped == 2
    assert meter.count == 0


def test_reset_clears_the_skipped_count():
    meter = AverageMeter()
    meter.update(float("inf"))
    meter.reset()
    assert meter.skipped == 0
