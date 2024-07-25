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
        save_steps,
        MODEL_SAVE_PATH,
    ):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.seed = seed
        self.epochs = epochs
        self.run = run
        self.save_steps = save_steps
        self.MODEL_SAVE_PATH = MODEL_SAVE_PATH

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

                    tag_loss = self.model(**batch).loss

                    loss_train += tag_loss.item() * batch["labels"].shape[0]

                    self.accelerator.backward(tag_loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()

                    self.scheduler.step()

                    step += 1

                    if step % self.save_steps == 0:
                        self.save_model(
                            self.seed, i, step, REPO_HOME, self.MODEL_SAVE_PATH
                        )

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

        self.save_model(self.seed, i, step, REPO_HOME, self.MODEL_SAVE_PATH)

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

    def save_model(self, seed, epoch, step, REPO_HOME, MODEL_SAVE_PATH):
        self.model.save_pretrained(
            os.path.join(
                REPO_HOME,
                MODEL_SAVE_PATH,
                "seed-" + str(seed) + "-epoch-" + str(epoch) + "-step-" + str(step),
            )
        )
