try:
    import objgraph
except ImportError:
    pass

import gc

from pytorch_lightning.callbacks import Callback


class GCCallback(Callback):
    def __init__(self, interval=1000):
        self.interval = interval
        self.next_print = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.next_print == 0:
            self.next_print = self.interval
            gc.collect()
        self.next_print -= 1


class MemoryLeakDegubber(Callback):
    def __init__(self, interval=1000):
        self.interval = interval
        self.next_print = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.next_print == 0:
            self.next_print = self.interval
            objgraph.show_most_common_types(limit=20)
        self.next_print -= 1
