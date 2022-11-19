# CW-provided functions
import evalutation
import dataset

import numpy as np
import time
from typing import Union, NamedTuple
import argparse
from multiprocessing import cpu_count
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchinfo import summary
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(
    description="Train a shallow CNN on GTZAN",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--log-dir",
    default=Path("logs"),
    type=Path)

parser.add_argument(
    "--learning-rate",
    default=5e-5,
    type=float,
    help="Learning rate"
)
parser.add_argument(
    "--batch-size",
    default=10,
    type=int,
    help="Number of audios within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=5,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=1,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--dropout",
    default=0.1,
    type=int,
    help="During training, randomly zeroes some of the elements of the input tensor \
          with probability p using samples from a Bernoulli distribution.",
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class InputShape(NamedTuple):
    height: int
    width: int
    channels: int


def main(args):
    # transform = transforms.ToTensor()

    # ================================================
    #                Loading data
    # ================================================
    train_dataset = dataset.GTZAN("../data/train.pkl")
    # "val_dataset": Testing && Validation data only used for assessing generization capability
    # of the model and tuning hyper-parameters
    val_dataset = dataset.GTZAN("../data/val.pkl")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        # pin_memory=True,
        # num_workers=args.worker_count,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        # num_workers=args.worker_count,
        # pin_memory=True,
    )

    sha_model = SHA_CNN(height=80, width=80, channels=1,
                        class_count=10, dropout_rate=args.dropout)
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # "using the stochastic Adam optimization [KB14] beta1 =0.9, beta2 =0.999,
    # epsilon=1e−08 and a learning rate of 0.00005."
    optimizer = torch.optim.Adam(
        sha_model.parameters(), betas=(0.9, 0.999), eps=1e-08, lr=args.learning_rate)

    # ================================================
    #                Tensorboard stuff
    # ================================================

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
        str(log_dir),
        flush_secs=5
    )

    sha_trainer = Trainer(
        sha_model, train_loader, val_loader, criterion,
        optimizer,
        summary_writer,
        DEVICE
    )

    sha_trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )


# * early imlementation of a sequential architecture,
# * will need to change to PARALLELISM with PIPELINE

class SHA_CNN(nn.Module):
    def __init__(
        self,
        dropout_rate,
        height: int = 80,
        width: int = 80,
        channels: int = 1,
        # 10 music genres
        class_count: int = 10,
    ):
        super().__init__()

        self.input_shape = InputShape(
            height=height, width=width, channels=channels)
        self.class_count = class_count

        # ===========================================================================
        #                    Time architecture (left-pipeline)
        # ===========================================================================

        #  conv1 = 16 temporal filters of shape(10, 23)
        self.conv1_l = nn.Conv2d(
            in_channels=self.input_shape.channels,
            # out_channels = how many conv-filters at this layer
            out_channels=16,
            kernel_size=(10, 23),
            padding="same",
        )
        self.initialise_layer(self.conv1_l)
        # Add a layer for batch-normalization after convolution and before activation
        self.norm1_l = nn.BatchNorm2d(self.conv1_l.out_channels)
        self.actv1_l = nn.LeakyReLU()
        self.pool1_l = nn.MaxPool2d(kernel_size=(1, 20))

        # ===========================================================================
        #                    Frequency architecture (right-pipeline)
        # ===========================================================================

        #  conv1 = 16 temporal filters of shape(21, 20)
        self.conv1_r = nn.Conv2d(
            in_channels=self.input_shape.channels,
            # out_channels = how many conv-filters at this layer
            out_channels=16,
            kernel_size=(21, 20),
            padding="same",
        )
        self.initialise_layer(self.conv1_r)
        # Add a layer for batch-normalization after convolution and before activation
        self.norm1_r = nn.BatchNorm2d(self.conv1_r.out_channels)
        self.actv1_r = nn.LeakyReLU()
        self.pool1_r = nn.MaxPool2d(kernel_size=(20, 1))

        # ===========================================================================
        #                           Merged architecture
        # ===========================================================================

        self.drop_m = nn.Dropout(dropout_rate)
        # "... the shape of 1×10240, which serves as input to a 200 units
        # fully connected layer with a dropout of 10%."
        self.fc_m = nn.Linear(10240, 200)
        self.initialise_layer(self.fc_m)

        # "All networks were trained towards
        # categorical-crossentropy objective using the stochastic
        # Adam optimization [KB14] with beta1 =0.9, beta2 =0.999,
        # epsilon=1e−08 and a learning rate of 0.00005."
        self.actv_m = nn.Softmax()

    def forward(self, audios: torch.Tensor) -> torch.Tensor:
        x_l = self.conv1_l(audios)
        x_l = self.norm1_l(x_l)
        x_l = self.actv1_l(x_l)
        x_l = self.pool1_l(x_l)

        x_r = self.conv1_r(audios)
        x_r = self.norm1_r(x_r)
        x_r = self.actv1_r(x_r)
        x_r = self.pool1_r(x_r)

        # The 16 feature maps of each pipeline are flattened
        # to a shape of 1×5120 and merged by concatenation into
        # the shape of 1×10240
        x_l = torch.flatten(x_l, start_dim=1)
        x_r = torch.flatten(x_r, start_dim=1)

        # ========================Layers after Merge============================
        x = torch.cat((x_l, x_r), 1)
        # Apply dropout to inputs of the fully-connected layer (before)
        x = self.drop_m(x)
        x = self.fc_m(x)
        x = self.actv_m(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        # self.model = model.to(device)
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            # ==============================================================
            # names: The filename which a given spectrogram belongs to
            # batch: The audio spectrogram, which relates to a randomly selected
            #        0.93 seconds of audio from "filename". The spectrograms are of size: [1, 80, 80].
            # labels: The class/label of the audio file
            # audios: The audio samples used to create the spectrogram
            # ==============================================================
            for names, batch, labels, audios in self.train_loader:
                data_load_end_time = time.time()
                logits = self.model.forward(batch)

                # TODO: BACK-PROPOGATION
                # Compute the loss using self.criterion and
                # store it in a variable called `loss`
                loss = self.criterion(logits, labels)
                # Compute the backward pass
                loss.backward()
                # Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time

                # if ((self.step + 1) % log_frequency) == 0:
                #     self.log_metrics(epoch, accuracy, loss,
                #                      data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss,
                                       data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"batch accuracy: {accuracy * 100:2.2f}, "
            f"data load time: "
            f"{data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for _, batch, labels, _ in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
            "accuracy",
            {"test": accuracy},
            self.step
        )
        self.summary_writer.add_scalars(
            "loss",
            {"test": average_loss},
            self.step
        )

        print(
            f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}"
        )


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
        f"CNN_bn_"
        f"bs={args.batch_size}_"
        f"lr={args.learning_rate}_"
        # f"brightness={args.data_aug_brightness}_" +
        # ("hflip_" if args.data_aug_hflip else "") +
        f"run_"
    )

    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())
