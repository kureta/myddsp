from typing import Tuple

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from torch import Tensor
from torch.utils.data import DataLoader

from myddsp.vae import Decoder, ZakDataset


def safe_log(x: Tensor, eps: float = 1e-5) -> Tensor:
    return torch.log(torch.clamp(x, min=eps))


class Zak(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = Decoder()
        self.dataset = ZakDataset()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=32, shuffle=True, num_workers=8)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.decoder.parameters(), lr=1e-4)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], _: int) -> Tensor:
        stft, amp, pitch = batch
        x_hat = self.decoder(amp, pitch)

        lin_loss = torch.nn.functional.mse_loss(x_hat, stft)
        log_loss = torch.nn.functional.mse_loss(safe_log(x_hat), safe_log(stft))
        loss = lin_loss + log_loss

        self.log("loss", loss, prog_bar=True)

        return loss


def main() -> None:
    model = Zak()
    trainer = pl.Trainer(
        callbacks=[pl_callbacks.RichProgressBar()],
        logger=pl_loggers.TensorBoardLogger("logs/", None),
        max_epochs=500,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
