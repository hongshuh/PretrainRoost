import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, NLLLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from roost.core_pretrain import Normalizer, RobustL1Loss, RobustL2Loss, sampled_softmax, BarlowTwinsLoss,BT_Magpie


def init_model(
    model_class,
    model_name,
    loss_params,
    model_params,
    run_id,
    optim,
    learning_rate,
    weight_decay,
    momentum,
    device,
    milestones=[],
    gamma=0.3,
    resume=None,
    fine_tune=None,
    transfer=None
):

    robust = model_params["robust"]
    # n_targets = model_params["n_targets"]
    #print(fine_tune)
    if fine_tune is not None:
        print(f"Use material_nn and output_nn from '{fine_tune}' as a starting point")
        checkpoint = torch.load(fine_tune, map_location=device)

        # update the task disk to fine tuning task
        checkpoint["model_params"]["task_dict"] = model_params["task_dict"]

        model = model_class(
            **checkpoint["model_params"],
            device=device,
        )
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        # model.trunk_nn.reset_parameters()
        # for m in model.output_nns:
        #     m.reset_parameters()

        assert model.model_params["robust"] == robust, (
            "cannot fine-tune "
            "between tasks with different numbers of outputs - use transfer "
            "option instead"
        )
        # assert model.model_params["n_targets"] == n_targets, (
        #     "cannot fine-tune "
        #     "between tasks with different numbers of outputs - use transfer "
        #     "option instead"
        # )

    elif transfer is not None:
        print(
            f"Use material_nn from '{transfer}' as a starting point and "
            "train the output_nn from scratch"
        )
        checkpoint = torch.load(transfer, map_location=device)

        model = model_class(device=device, **model_params)
        model.to(device)

        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["state_dict"].items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    elif resume:
        # TODO work out how to ensure that we are using the same optimizer
        # when resuming such that the state dictionaries do not clash.
        print(f"Resuming training from '{resume}'")
        checkpoint = torch.load(resume, map_location=device)

        model = model_class(
            **checkpoint["model_params"],
            device=device,
        )
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.epoch = checkpoint["epoch"]
        model.best_val_score = checkpoint["best_val_score"]

    else:
        model = model_class(device=device, **model_params)

        model.to(device)

    # Select Optimiser
    print(optim)
    if optim == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optim == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optim == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise NameError("Only SGD, Adam or AdamW are allowed as --optim")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )

    if resume:
        # NOTE the user could change the optimizer when resuming creating a bug
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    print(f"Total Number of Trainable Parameters: {model.num_params:,}")

    # TODO parallelise the code over multiple GPUs. Currently DataParallel
    # crashes as subsets of the batch have different sizes due to the use of
    # lists of lists rather the zero-padding.
    # if (torch.cuda.device_count() > 1) and (device==torch.device("cuda")):
    #     print("The model will use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    print(device)

    model.to(device)

    return model, optimizer, scheduler


def init_losses(task_dict, loss_dict, loss_params, robust=False):  # noqa: C901

    batch_size = loss_params['batch_size']
    embed_size = loss_params['embed_size']
    print('embed_size = ', embed_size)
    lambda_    = loss_params['lambda_'] 
    print('lambda_ = ', lambda_)
     
    criterion_dict = {}
    #print(task_dict.items())
    print("Loss",loss_dict)
    for name, task in task_dict.items():
        # Select Task and Loss Function
        # print("name",name)
        # print("task",task)
        if task == "SSL":
            if loss_dict[name] == "SSL":
                criterion_dict[name] = (task, BarlowTwinsLoss("cpu",batch_size,embed_size,lambda_))
    # print(criterion_dict.items())
    print(criterion_dict)
    return criterion_dict


def train_ensemble(
    model_class,
    model_name,
    run_id,
    ensemble_folds,
    epochs,
    patience,
    train_set,
    val_set,
    log,
    data_params,
    setup_params,
    restart_params,
    model_params,
    loss_params,
    loss_dict,
    log_dir
):
    """
    Train multiple models
    """

    train_generator = DataLoader(train_set, **data_params)
    #print(train_generator)

    if val_set is not None:
        data_params.update({"batch_size": 16 * data_params["batch_size"]})
        val_generator = DataLoader(val_set, **data_params)
    else:
        val_generator = None

    #print(ensemble_folds)
    for j in range(ensemble_folds):
        #print(j)
        #  this allows us to run ensembles in parallel rather than in series
        #  by specifying the run-id arg.
        if ensemble_folds == 1:
            j = run_id

        model, optimizer, scheduler = init_model(
            model_class=model_class,
            model_name=model_name,
            model_params=model_params,
            loss_params= loss_params,
            run_id=j,
            **setup_params,
            **restart_params,
        )

        criterion_dict = init_losses(model.task_dict, loss_dict, loss_params, model_params["robust"])

        if log:
            writer = SummaryWriter(
                log_dir=(
                    f"runs/{model_name}/{model_name}-r{j}_{datetime.now():%d-%m-%Y_%H-%M-%S}"
                )
            )
        else:
            writer = None

        if (val_set is not None) and (model.best_val_scores is None):
            print("Getting Validation Baseline")
            with torch.no_grad():
                v_metrics = model.evaluate(
                    generator=val_generator,
                    criterion_dict=criterion_dict,
                    optimizer=None,
                    action="val",
                )

                val_score = {}

                for name, task in model.task_dict.items():
                    if task == "regression":
                        val_score[name] = v_metrics[name]["MAE"]
                        print(
                            f"Validation Baseline - {name}: MAE {val_score[name]:.3f}"
                        )
                    elif task == "classification":
                        val_score[name] = v_metrics[name]["Acc"]
                        print(
                            f"Validation Baseline - {name}: Acc {val_score[name]:.3f}"
                        )
                    elif task == 'SSL':
                        val_score[name] = v_metrics[name]["Loss"]
                        print(
                            f"Validation Baseline - {name}: Loss {val_score[name]:.3f}"
                        )
                model.best_val_scores = val_score
        # print("task is ",task)
        # print(v_metrics[name])
        torch.save(model.state_dict(), os.path.join(log_dir, 'checkpoint.pth.tar'))
        torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth.tar'))
        model.fit(
            train_generator=train_generator,
            val_generator=val_generator,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            criterion_dict=criterion_dict,
            model_name=model_name,
            run_id=j,
            writer=writer,
            patience=patience,
            log_dir=log_dir
        )

        # torch.save(model.state_dict(), os.path.join(log_dir, 'model.pth'))
        # torch.save(model, os.path.join(log_dir, 'model_.pth'))
        #print("Hi there")

