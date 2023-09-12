import gc
import json
import shutil
from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import softmax
from tqdm.autonotebook import tqdm


class BaseModelClass(nn.Module, ABC):
    """
    A base class for models.
    """

    def __init__(self, task_dict, robust, device, epoch=1, best_val_scores=None):
        """
        Args:
            task (str): "regression" or "classification"
            robust (bool): whether an aleatoric loss function is being used
            device (pytorch.device): the device the model will be run on
            epoch (int): the epoch model training will begin/resume from
            best_val_score (float): validation score to use for early stopping
        """
        super().__init__()
        self.task_dict = task_dict
        #self.target_names = list(task_dict.keys())
        self.robust = robust
        self.device = device
        self.epoch = epoch
        self.best_val_scores = best_val_scores
        self.es_patience = 0

        self.model_params = {"task_dict": task_dict}

    def fit(  # noqa: C901
        self,
        train_generator,
        val_generator,
        optimizer,
        scheduler,
        epochs,
        criterion_dict,
        model_name,
        run_id,
        checkpoint=True,
        writer=None,
        verbose=True,
        patience=None,
        log_dir=None
    ):
        """
        Args:

        """
        start_epoch = self.epoch

        try:
            for epoch in range(start_epoch, start_epoch + epochs):
                self.epoch += 1
                # Training
                t_metrics = self.evaluate(
                    generator=train_generator,
                    criterion_dict=criterion_dict,
                    optimizer=optimizer,
                    action="train",
                    verbose=verbose,
                )

                if writer is not None:
                    for task, metrics in t_metrics.items():
                        for metric, val in metrics.items():
                            writer.add_scalar(f"{task}/train/{metric}", val, epoch)

                if verbose:
                    print(f"Epoch: [{epoch}/{start_epoch + epochs - 1}]")
                    for task, metrics in t_metrics.items():
                        print(
                            f"Train \t\t: {task} - "
                            + "".join(
                                [f"{key} {val:.3f}\t" for key, val in metrics.items()]
                            )
                        )

                # Validation
                is_best = False
                if val_generator is not None:
                    with torch.no_grad():
                        # evaluate on validation set
                        v_metrics = self.evaluate(
                            generator=val_generator,
                            criterion_dict=criterion_dict,
                            optimizer=None,
                            action="val",
                            verbose=False,
                        )

                    if writer is not None:
                        for task, metrics in v_metrics.items():
                            for metric, val in metrics.items():
                                writer.add_scalar(
                                    f"{task}/validation/{metric}", val, epoch
                                )

                    if verbose:
                        for task, metrics in v_metrics.items():
                            print(
                                f"Validation \t: {task} - "
                                + "".join(
                                    [
                                        f"{key} {val:.3f}\t"
                                        for key, val in metrics.items()
                                    ]
                                )
                            )

                    # TODO test all tasks to see if they are best,
                    # save a best model if any is best.
                    # TODO what are the costs of this approach.
                    # It could involve saving a lot of models?

                    is_best = []
                    
                    

                    for name in self.best_val_scores:
                        if self.task_dict[name] == "SSL":
                            # print('find best here')
                            if v_metrics[name]["Loss"] < self.best_val_scores[name]:
                                self.best_val_scores[name] = v_metrics[name]["Loss"]
                                is_best.append(True)
                            is_best.append(False)
                    # print('best_val_score',self.best_val_scores)

                    if any(is_best):
                        self.es_patience = 0
                    else:
                        self.es_patience += 1
                        if patience:
                            if self.es_patience > patience:
                                print(
                                    "Stopping early due to lack of improvement on Validation set"
                                )
                                break

                if checkpoint:
                    checkpoint_dict = {
                        "model_params": self.model_params,
                        "state_dict": self.state_dict(),
                        "epoch": self.epoch,
                        "best_val_score": self.best_val_scores,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }

                    # TODO saving a model at each epoch may be slow?
                    # save_checkpoint(checkpoint_dict, False, model_name, run_id)

                    # TODO when to save best models? should this be done task-wise in
                    # the multi-task case?
                    # print("what is 'is best'",is_best)
                    save_checkpoint(checkpoint_dict, is_best, log_dir, run_id)

                scheduler.step()

                # catch memory leak
                gc.collect()

        except KeyboardInterrupt:
            pass

        if writer is not None:
            writer.close()

    def evaluate(
        self,
        generator,
        criterion_dict,
        optimizer,
        action="train",
        verbose=False,
    ):
        """
        evaluate the model
        """

        if action == "val":
            self.eval()
        elif action == "train":
            self.train()
        else:
            raise NameError("Only train or val allowed as action")

        metrics = {
            key: {k: [] for k in ["Loss"]}
            for key in self.task_dict
        }

        # we do not need batch_comp or batch_ids when training
        # disable output in non-tty (e.g. log files) https://git.io/JnBOi
        for inputs_1, magpie_feature,*_ in tqdm(
            generator, disable=True if not verbose else None
        ):
            # move tensors to GPU
            inputs_1 = (tensor.to(self.device) for tensor in inputs_1)

            # print(f'magpie feature shape is {magpie.shape}')

            # compute output
            outputs_1 = self(*inputs_1)

            # print(outputs_1.shape)
            #print(outputs_2.shape)
            task, criterion = criterion_dict["Pre"]
            # print(outputs_1.shape)
            # print(magpie_feature.shape)
            # exit()
            magpie_feature = magpie_feature.to(self.device)
            loss = criterion.forward(outputs_1,magpie_feature)
            #print(loss)
            #print(metrics)

            if task == "SSL":
                metrics["Pre"]["Loss"].append(loss.cpu().item())

                # NOTE we are currently just using a direct sum of losses
                # this should be okay but is perhaps sub-optimal

            if action == "train":
                # compute gradient and take an optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics = {
            key: {k: np.array(v).mean() for k, v in d.items() if v}
            for key, d in metrics.items()
        }
        #print(metrics)
        return metrics

    @torch.no_grad()
    def featurise(self, generator):
        """Generate features for a list of composition strings. When using Roost,
        this runs only the message-passing part of the model without the ResNet.

        Args:
            generator (DataLoader): PyTorch loader with the same data format used in fit()

        Returns:
            np.array: 2d array of features
        """
        err_msg = f"{self} needs to be fitted before it can be used for featurisation"
        assert self.epoch > 0, err_msg

        self.eval()  # ensure model is in evaluation mode
        features_1 = []
        features_2 = []

        for input_1, input_2,  *_ in generator:

            input_1 = (tensor.to(self.device) for tensor in input_1)

            output_1 = self.trunk_nn(self.material_nn(*input_1)).cpu().numpy()
            features_1.append(output_1)


            input_2 = (tensor.to(self.device) for tensor in input_2)

            output_2 = self.trunk_nn(self.material_nn(*input_2)).cpu().numpy()
            features_2.append(output_2)

        return np.vstack(features_1), np.vstack(features_2)

    @abstractmethod
    def forward(self, *x):
        """
        Forward pass through the model. Needs to be implemented in any derived
        model class.
        """
        raise NotImplementedError("forward() is not defined!")

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.tensor(0)
        self.std = torch.tensor(1)

    def fit(self, tensor, dim=0, keepdim=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor, dim, keepdim)
        self.std = torch.std(tensor, dim, keepdim)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()

    @classmethod
    def from_state_dict(cls, state_dict):
        instance = cls()
        instance.mean = state_dict["mean"].cpu()
        instance.std = state_dict["std"].cpu()

        return instance


class Featurizer:
    """Base class for featurizing nodes and edges."""

    def __init__(self, allowed_types):
        self.allowed_types = allowed_types
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types, f"{key} is not an allowed atom type"
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = self._embedding.keys()

    def get_state_dict(self):
        return self._embedding

    @property
    def embedding_size(self):
        return len(list(self._embedding.values())[0])

    @classmethod
    def from_json(cls, embedding_file):
        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = embedding.keys()
        # print(allowed_types)
        instance = cls(allowed_types)
        for key, value in embedding.items():
            #print(key)
            instance._embedding[key] = np.array(value, dtype=float)
        
        #print(dir(instance))
        return instance


def save_checkpoint(state, is_best, log_dir, run_id):
    """
    Saves a checkpoint and overwrites the best model when is_best = True
    """

    checkpoint = f"{log_dir}/checkpoint.pth.tar"
    best = f"{log_dir}/best.pth.tar"

    torch.save(state, checkpoint)
    if any(is_best):
        print('better model than before')
        shutil.copyfile(checkpoint, best)


def RobustL1Loss(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = np.sqrt(2.0) * torch.abs(output - target) * torch.exp(-log_std) + log_std
    return torch.mean(loss)


def RobustL2Loss(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    # NOTE can we scale log_std by something sensible to improve the OOD behaviour?
    loss = 0.5 * torch.pow(output - target, 2.0) * torch.exp(-2.0 * log_std) + log_std
    return torch.mean(loss)


def sampled_softmax(pre_logits, log_std, samples=10):
    """
    Draw samples from gaussian distributed pre-logits and use these to estimate
    a mean and aleatoric uncertainty.
    """
    # NOTE here as we do not risk dividing by zero should we really be
    # predicting log_std or is there another way to deal with negative numbers?
    # This choice may have an unknown effect on the calibration of the uncertainties
    sam_std = torch.exp(log_std).repeat_interleave(samples, dim=0)

    # TODO here we are normally distributing the samples even if the loss
    # uses a different prior?
    epsilon = torch.randn_like(sam_std)
    pre_logits = pre_logits.repeat_interleave(samples, dim=0) + torch.mul(
        epsilon, sam_std
    )
    logits = softmax(pre_logits, dim=1).view(len(log_std), samples, -1)
    return torch.mean(logits, dim=1)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self,device,batch_size,embed_size,lambd):
        super(BarlowTwinsLoss,self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.lambd = lambd # default=0.005
        self.bn = nn.BatchNorm1d(self.embed_size, affine=False).to(self.device)

    def forward(self, z1, z2):

        # empirical cross-correlation matrix
        z1 = z1.to(self.device)
        z2 = z2.to(self.device)
        c = self.bn(z1).T @ self.bn(z2).to(self.device)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        #torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().to(self.device)
        off_diag = off_diagonal(c).pow_(2).sum().to(self.device)
        loss = on_diag + self.lambd * off_diag
        return loss.to(self.device)
    
class BT_Magpie(nn.Module):
    def __init__(self,device,batch_size,embed_size,lambd):
        super(BT_Magpie,self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.lambd = lambd # default=0.005
        self.bn = nn.BatchNorm1d(self.embed_size, affine=False).to(self.device)
        self.beta = 0.1 # hyperparameter for magpie

    def forward(self, z1, z2,magpie):

        # empirical cross-correlation matrix
        z1_unmasked = z1[0].to(self.device)
        z1_magpie = z1[1].to(self.device)
        z2 = z2[0].to(self.device)
        c = self.bn(z1_unmasked).T @ self.bn(z2).to(self.device)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        #torch.distributed.all_reduce(c)

        #Calculate loss for magpie
        magpie = magpie.to(self.device)
        magpie_MSE = nn.MSELoss()(magpie,z1_magpie)
        

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().to(self.device)
        off_diag = off_diagonal(c).pow_(2).sum().to(self.device)
        BT_loss = on_diag + self.lambd * off_diag
        magpie_loss = self.beta * magpie_MSE
        loss = BT_loss+magpie_loss
        print(f'magpie loss is {magpie_loss}')
        print(f'BT loss is {BT_loss}')

        return loss.to(self.device)
