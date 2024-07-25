# author: ddukic

import os

from constants import REPO_HOME
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        loader,
        optimizer,
        scheduler,
        accelerator,
        seed,
        epochs,
        run,
        from_pretrained_seed,
        from_pretrained_epoch,
        from_pretrained_step,
        MODEL_SAVE_PATH,
        unlock_config=None,
    ):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.seed = seed
        self.epochs = epochs
        self.run = run
        self.from_pretrained_seed = from_pretrained_seed
        self.from_pretrained_epoch = from_pretrained_epoch
        self.from_pretrained_step = from_pretrained_step
        self.MODEL_SAVE_PATH = MODEL_SAVE_PATH
        self.unlock_config = unlock_config

    def train(
        self,
    ):
        print("Training in progress...")

        print("=" * 30)

        for i in tqdm(range(self.epochs)):
            loss_train = 0.0

            self.model.train()

            step = 0

            for batch in self.loader:
                with self.accelerator.accumulate(self.model):
                    self.model.zero_grad()
                    self.optimizer.zero_grad()

                    if self.unlock_config is None:
                        tag_loss = self.model(**batch).loss
                    else:
                        tag_loss = self.model(
                            **batch, unlock_config=self.unlock_config
                        ).loss

                    loss_train += tag_loss.item() * batch["labels"].shape[0]

                    self.accelerator.backward(tag_loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()

                    self.scheduler.step()

                    step += 1

            if self.accelerator.is_local_main_process:
                self.logging(
                    loss=loss_train, loader=self.loader, epoch=i, normalized=False
                )

            # x axis goes from 0 to 9 because of wandb
            train_metrics = {
                "train/epoch_loss_seed_"
                + str(self.seed): loss_train / len(self.loader.dataset),
                "epoch": i,
            }

            self.run.log({**train_metrics})

        self.save_model(
            i,
            step,
            REPO_HOME,
        )

    def logging(self, loss, loader, epoch, mode="train", normalized=False):
        if normalized:
            loss_string = mode + " loss: " + str(round(loss, 3))
        else:
            loss_string = mode + " loss: " + str(round(loss / len(loader.dataset), 3))

        epoch_string = "Epoch: " + str(epoch)

        if mode == "train":
            print(epoch_string)
        print(loss_string)
        print("=" * 30)

    def save_model(
        self,
        epoch,
        step,
        REPO_HOME,
    ):
        self.model.save_pretrained(
            os.path.join(
                REPO_HOME,
                self.MODEL_SAVE_PATH,
                "seed-"
                + str(self.seed)
                + "-epoch-"
                + str(epoch)
                + "-step-"
                + str(step)
                + "-pretrseed-"
                + str(self.from_pretrained_seed)
                + "-pretrepoch-"
                + str(self.from_pretrained_epoch)
                + "-pretrstep-"
                + str(self.from_pretrained_step),
            )
        )
