import math


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.skipped = 0

    def update(self, val, n=1):
        # A non-finite value is dropped rather than accumulated. `sum` is a
        # running total, so one inf makes it inf forever: every later average
        # is inf too, and the loss chart in the UI reads "inf" for the rest of
        # a run that recovered on the next iteration. That is exactly what a
        # float16 gradient overflow produces, and it makes a healthy run look
        # broken long after the event.
        #
        # Counted rather than ignored, so the number is available to anyone
        # asking why an average covers fewer samples than it should.
        if not math.isfinite(val):
            self.skipped += n
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
