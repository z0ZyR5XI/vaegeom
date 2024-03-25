from logging import Logger

import lightning as L


class ProgressMsg(L.Callback):
    """
    Log message with training and validation loss to logger.
    """

    def __init__(
        self,
        logger: Logger,
        *args,
        log_interval: int = 1,
        **kwargs):
        self.logger = logger
        self.log_interval = log_interval

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        epoch = trainer.current_epoch
        #val_loss = trainer.callback_metrics['val_loss']
        if epoch % self.log_interval == 0:
            self.logger.info(f'Checkpoint')
        
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        val_loss = trainer.callback_metrics['val_loss']
        if epoch % self.log_interval == 0:
            self.logger.info(f'Epoch = {epoch}, val_loss = {val_loss :.4f}')
        