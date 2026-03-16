import torch
from torch import nn
from torch.utils.data import DataLoader
from clearml import Task

from pipelines.nn.encoder.encoder_model import WheelModel
from pipelines.nn.encoder.dataset import WheelSignalsDataset
from pipelines.nn.encoder.metrics import Metrics
from mixins.file_operator import FileOperator
import os
from config import Common, Paths
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

torch.manual_seed(Common.SEED)


class Trainer(FileOperator):
    def __init__(
        self,
        data,
        n_points: int = 1536,
        patch: int = 32,
        d_model: int = 128,
        lr: float = 0.001,
        batch_size: int = 32,
        dropout: float = 0.1,
    ):

        super().__init__()

        self.task = Task.init(project_name="wheel-defect", task_name="encoder-training")
        self.task.connect(
            {
                "lr": lr,
                "batch_size": batch_size,
                "dataset_size": len(data),
                "model": "WheelModel",
            }
        )

        self.logger = self.task.get_logger()
        self.data = data
        self.lr = lr
        self.batch_size = batch_size
        self.metrics = Metrics(self.logger)
        self.model = WheelModel(n_points, patch, d_model, dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")
        self.model.to(self.device)

    def train_val_split(self, ratio: float = 0.9):
        train_idx = int(len(self.data) * ratio)
        train = self.data[:train_idx]
        val = self.data[train_idx:]
        train_ds = WheelSignalsDataset(train)
        val_ds = WheelSignalsDataset(val)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def train_step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        fault, profile = self.model(batch["X"])

        loss_fault = self.loss_fn(fault, batch["y_fault"])
        loss_profile = self.loss_fn(profile, batch["y_profile"])
        loss = loss_fault + loss_profile

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_fault.item(), loss_profile.item()

    def train(self, epochs: int = 10, early_stopping_patience: int = 15, eval_every: int = 5):
        train_loader, val_loader = self.train_val_split()

        loss_tracker = {}
        stopper_counter = 0

        self.model.train()

        for epoch in tqdm(range(epochs), desc="training encoder"):
            batch_loss = []
            batch_fault_loss = []
            batch_profile_loss = []

            for batch in train_loader:
                loss, loss_fault, loss_profile = self.train_step(batch)
                batch_loss.append(loss)
                batch_fault_loss.append(loss_fault)
                batch_profile_loss.append(loss_profile)

            mean_loss = sum(batch_loss) / len(batch_loss)
            mean_fault_loss = sum(batch_fault_loss) / len(batch_fault_loss)
            mean_profile_loss = sum(batch_profile_loss) / len(batch_profile_loss)
            loss_tracker[epoch] = mean_loss

            self.logger.report_scalar("loss", "train_total", mean_loss, epoch)
            self.logger.report_scalar("loss", "train_fault", mean_fault_loss, epoch)
            self.logger.report_scalar("loss", "train_profile", mean_profile_loss, epoch)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"loss={mean_loss:.4f} | "
                f"fault={mean_fault_loss:.4f} | "
                f"profile={mean_profile_loss:.4f}"
            )

            if epoch % eval_every == 0 or epoch == epochs - 1:
                metrics = self.evaluate(val_loader)

                self.logger.report_scalar("loss", "val_fault", metrics["fault_loss"], epoch)
                self.logger.report_scalar("loss", "val_profile", metrics["profile_loss"], epoch)
                self.logger.report_scalar("acc", "val_fault", metrics["fault_acc"], epoch)
                self.logger.report_scalar("acc", "val_profile", metrics["profile_acc"], epoch)

                self.metrics.log_confusion_matrices(metrics, epoch)
                self.metrics.log_fault_pr_curve(metrics, epoch)
                self.metrics.log_profile_pr_curves(metrics, epoch, num_classes=3)

            if epoch > 0:
                if loss_tracker[epoch] > loss_tracker[epoch - 1]:
                    stopper_counter += 1
                    if stopper_counter >= early_stopping_patience:
                        print("Stopping training.")
                        break
                else:
                    stopper_counter = 0

        ckpt_dir = os.path.join(Paths.DATA, "nn_checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "encoder_model.pt")
        torch.save(self.model.state_dict(), ckpt_path)

        self.task.upload_artifact(name="encoder_model", artifact_object=ckpt_path)
        self.task.close()

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()

        all_y_fault = []
        all_y_profile = []

        all_fault_pred = []
        all_profile_pred = []

        all_fault_prob = []
        all_profile_prob = []

        total_fault_loss = 0.0
        total_profile_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            fault_logits, profile_logits = self.model(batch["X"])

            loss_fault = self.loss_fn(fault_logits, batch["y_fault"])
            loss_profile = self.loss_fn(profile_logits, batch["y_profile"])

            fault_probs = torch.softmax(fault_logits, dim=1)
            profile_probs = torch.softmax(profile_logits, dim=1)

            fault_pred = torch.argmax(fault_probs, dim=1)
            profile_pred = torch.argmax(profile_probs, dim=1)

            all_y_fault.append(batch["y_fault"].cpu().numpy())
            all_y_profile.append(batch["y_profile"].cpu().numpy())

            all_fault_pred.append(fault_pred.cpu().numpy())
            all_profile_pred.append(profile_pred.cpu().numpy())

            all_fault_prob.append(fault_probs.cpu().numpy())
            all_profile_prob.append(profile_probs.cpu().numpy())

            total_fault_loss += loss_fault.item()
            total_profile_loss += loss_profile.item()
            n_batches += 1

        y_fault = np.concatenate(all_y_fault)
        y_profile = np.concatenate(all_y_profile)

        fault_pred = np.concatenate(all_fault_pred)
        profile_pred = np.concatenate(all_profile_pred)

        fault_prob = np.concatenate(all_fault_prob)  # (N, 2)
        profile_prob = np.concatenate(all_profile_prob)  # (N, C)

        metrics = {
            "fault_loss": total_fault_loss / n_batches,
            "profile_loss": total_profile_loss / n_batches,
            "fault_acc": accuracy_score(y_fault, fault_pred),
            "profile_acc": accuracy_score(y_profile, profile_pred),
            "y_fault": y_fault,
            "y_profile": y_profile,
            "fault_pred": fault_pred,
            "profile_pred": profile_pred,
            "fault_prob": fault_prob,
            "profile_prob": profile_prob,
        }

        self.model.train()
        return metrics
