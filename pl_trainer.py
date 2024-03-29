import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from unet import UNet


class NaturalnessTrainer(pl.LightningModule):
    """
    A simplified UNet model for interpreting modality influence via occlusion sensitivity.
    """

    def __init__(
        self,
        n_input_ch,
        input_size,
        learning_rate,
        batch_size,
        use_lr_scheduler,
        log_distributions_flag,
        exp_name,
        occlusion_type,
        use_embedding_occlusion="test",
    ):
        super().__init__()

        self.verbose = False
        self.fill_value = 0
        self.occlusion_type = occlusion_type

        self.use_embedding_occlusion_train = (
            True if use_embedding_occlusion == "train" else None
        )
        self.use_embedding_occlusion_test = (
            True if use_embedding_occlusion == "test" else None
        )
        self.exp_name = exp_name

        self.input_size = input_size

        self.train_log_diffs = []
        self.test_log_diffs = []

        # self.loss = CustomLoss(mode='mse', ignore_nan=True)  # or mode='mae'
        self.loss = CustomLoss(mode="mae", ignore_nan=True)  # or mode='mae'
        self.custom_mae = CustomMAE()
        self.custom_rmse = CustomRMSE()

        self.log_distributions_flag = log_distributions_flag

        self.learning_rate = learning_rate
        self.weight_decay = 1e-3
        self.use_lr_scheduler = use_lr_scheduler

        self.model = UNet(
            input_ch=n_input_ch, patch_size=self.input_size, batch_size=batch_size
        )

        # Modality mapping -lookup table
        self.modality_indices_train = {
            "S2": list(range(0, 10)),
            "S1": [10, 11],
            "WC": [12],
            "VIIRS": [13],
        }

        # Modality mapping -lookup table
        self.modality_indices_test = {
            "S2": list(range(0, 10)),
            "S1": [10, 11],
            "WC": [12],
            "VIIRS": [13],
            "ALL": list(range(0, 14)),
        }

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        sample, wdpa_id = batch
        image = sample["image"]
        mask = sample["mask"]
        forward_dict = self(image)
        loss = self.loss(forward_dict["pred_act"], mask)

        diffs_per_modality = {}

        if self.use_embedding_occlusion_train:
            # Calculate regularization loss only if the flag is enabled
            regularization_loss = 0.0
            # print("IMAGE SHAPE", image.shape)
            original_embedding = self.model.forward(
                image, encode_only=True
            )  # image: 8,14,256,256, orig_embed: 8,512,1616,1
            # print("ORIGINAL EMBEDDING", original_embedding.shape)

            diffs_per_modality = {
                modality: None for modality in self.modality_indices_train
            }

            # Iterate over each modality to occlude its channels and generate an embedding
            for modality, indices in self.modality_indices_train.items():
                occluded_x = self._generate_occluded_input(image, indices)
                occluded_embedding = self.model.forward(occluded_x, encode_only=True)
                diff = original_embedding - occluded_embedding  # diff 8, 512, 16, 16, 1
                # print("DIFF SHAPE", diff.shape)
                norm = torch.norm(
                    diff, p=2, dim=(1, 2, 3, 4), keepdim=True
                ).squeeze()  # L2 norm on batch level. norm is a scalar
                # print("norm", norm.shape)

                diffs_per_modality[modality] = (
                    norm.detach()
                )  # Mean across batch and detach to remove from computation graph

                regularization_loss += norm.mean()

            # Average the regularization loss over the number of modalities
            regularization_loss /= len(self.modality_indices_train)
            lambda_reg = 0.001  # Regularization strength
            total_loss = loss + lambda_reg * regularization_loss

            batch_diffs = self.log_diffs(wdpa_id, diffs_per_modality)
            self.train_log_diffs.extend(batch_diffs)

        else:
            # If regularization is not used, the total loss is just the task loss
            total_loss = loss

        self.log(
            "train_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if self.use_embedding_occlusion_train:
            # Only log these if regularization is used
            self.log(
                "train_reg_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "train_embed_loss",
                regularization_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        self.log(
            "train_mae",
            self.custom_mae(forward_dict["pred_act"], mask),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_rmse",
            self.custom_rmse(forward_dict["pred_act"], mask),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        sample, wdpa_id = batch
        image = sample["image"]
        mask = sample["mask"]
        forward_dict = self(image)

        loss = self.loss(forward_dict["pred_act"], mask)
        mae = self.custom_mae(forward_dict["pred_act"], mask)
        rmse = self.custom_rmse(forward_dict["pred_act"], mask)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        sample, wdpa_id = batch
        image = sample["image"]
        mask = sample["mask"]
        forward_dict = self(image)

        loss = self.loss(forward_dict["pred_act"], mask)
        mae = self.custom_mae(forward_dict["pred_act"], mask)
        rmse = self.custom_rmse(forward_dict["pred_act"], mask)

        if self.use_embedding_occlusion_test:
            original_embedding = self.model.forward(image, encode_only=True)
            diffs_per_modality = {}

            # Iterate over each modality to occlude its channels and generate an embedding
            for modality, indices in self.modality_indices_test.items():
                occluded_x = self._generate_occluded_input(image, indices)
                occluded_embedding = self.model.forward(occluded_x, encode_only=True)
                diff = original_embedding - occluded_embedding
                norm = torch.norm(
                    diff, p=2, dim=(1, 2, 3), keepdim=True
                ).squeeze()  # L2 norm on batch level

                diffs_per_modality[modality] = (
                    norm.detach()
                )  # Detach to remove from computation graph

            # Log diffs
            batch_diffs = self.log_diffs(wdpa_id, diffs_per_modality)
            self.test_log_diffs.extend(batch_diffs)

        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def on_test_end(self):
        if self.use_embedding_occlusion_test:
            # Save dictionaries containing diffs and wdpa_ids
            print("USE REG TEST")
            torch.save(
                self.test_log_diffs,
                f"./logs/{self.exp_name}/logged_test_diffs_{self.exp_name}.pt",
            )

    def on_train_end(self):
        if self.use_embedding_occlusion_train:
            # Save dictionaries containing diffs and wdpa_ids
            torch.save(
                self.train_log_diffs,
                f"./logs/{self.exp_name}/logged_diffs_{self.exp_name}.pt",
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.use_lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer, T_0=3, T_mult=1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
            }
        else:
            return optimizer

    def _generate_occluded_input(self, x, modality_indices):
        """
        Occludes the channels for a given modality with different strategies.
        """
        # Clone the input to avoid modifying the original tensor
        occluded_x = x.clone()

        for idx in modality_indices:
            if self.verbose:
                # Log the mean of the band before occlusion for sanity check
                original_mean = occluded_x[:, idx, :, :].mean().item()
                print(f"original_mean_band_{idx}", original_mean)

            # Apply the specified occlusion strategy
            if self.occlusion_type == "zero":
                print("zero")
                occluded_x[:, idx, :, :] = 0
            elif self.occlusion_type == "one":
                print("one")
                occluded_x[:, idx, :, :] = 1
            elif self.occlusion_type == "random":
                print("random")
                occluded_x[:, idx, :, :] = torch.rand(occluded_x[:, idx, :, :].shape)
            elif self.occlusion_type == "gaussian":
                print("gaussian")
                occluded_x[:, idx, :, :] = torch.randn(occluded_x[:, idx, :, :].shape)
            else:
                raise ValueError("Unsupported occlusion type specified")

            if self.verbose:
                # Log the mean of the band after occlusion for sanity check
                occluded_mean = occluded_x[:, idx, :, :].mean().item()
                print(f"occluded_mean_band_{idx}", occluded_mean)

        return occluded_x

    def log_diffs(self, wdpa_id, diffs_per_modality):  # TODO NOT WORKING YET
        # Ensure we have a storage structure for the logged diffs
        logged_diffs = []

        # Prepare a record for the current batch's diffs and corresponding wdpa_id
        diff_record = {
            "wdpa_id": wdpa_id.detach().cpu(),  # Detach and move to CPU for logging/saving
            "diffs": {
                modality: diff.detach().cpu()
                for modality, diff in diffs_per_modality.items()
            },
        }

        # Append the record to the list of logged diffs
        logged_diffs.append(diff_record)
        return logged_diffs


class CustomRMSE(nn.Module):
    def __init__(self):
        super(CustomRMSE, self).__init__()

    def forward(self, pred, target):
        valid_mask = ~torch.isnan(target)
        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]
        mse = torch.square(valid_pred - valid_target).mean()
        rmse = torch.sqrt(mse)
        return rmse


class CustomMAE(nn.Module):
    def __init__(self):
        super(CustomMAE, self).__init__()

    def forward(self, pred, target):
        valid_mask = ~torch.isnan(target)
        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]
        mae = torch.abs(valid_pred - valid_target).mean()
        return mae


class CustomLoss(torch.nn.Module):
    def __init__(self, mode="mse", ignore_nan=True):
        """
        Initializes the custom loss module.

        Parameters:
        - mode (str): 'mse' for Mean Squared Error, 'mae' for Mean Absolute Error.
        - ignore_nan (bool): If True, NaN values in the target will be ignored.
        """
        super().__init__()
        assert mode in ["mse", "mae"], "mode must be 'mse' or 'mae'"
        self.mode = mode
        self.ignore_nan = ignore_nan

    def forward(self, input, target):
        """
        Calculates the loss between input and target.

        Parameters:
        - input (Tensor): Predictions from the model.
        - target (Tensor): Ground truth values.

        Returns:
        - Tensor: Calculated loss.
        """
        if self.ignore_nan:
            valid_mask = ~torch.isnan(target)
            input = input[valid_mask]
            target = target[valid_mask]
        else:
            input = torch.nan_to_num(input)
            target = torch.nan_to_num(target)

        if self.mode == "mse":
            return F.mse_loss(input, target, reduction="mean")
        else:
            return F.l1_loss(input, target, reduction="mean")
