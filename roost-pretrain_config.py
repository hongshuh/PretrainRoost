import argparse
import os
import sys
import yaml
import torch
from sklearn.model_selection import train_test_split as split

from roost.roost.data_pretrain_fl_mml import CompositionData, collate_batch
from roost.roost.model_pretrain_fl_mml import Roost
from roost.utils_pretrain_fl_mml import  train_ensemble
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
import sys
import time


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class AgnosticSL(object):
    def __init__(self, config):
        self.config = config
        self.device = self.config['device']
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time
        self.log_dir = os.path.join('runs_contrast', dir_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def pretrain(self):

        data_path = self.config['data_path']
        fea_path = self.config['fea_path']
        tasks = self.config['tasks']
        targets = self.config['targets']
        losses = self.config['losses']
        robust = self.config['robust']
        data_id =self.config['data_id']
        model_name = self.config['model_name']
        elem_fea_len = self.config['elem_fea_len']
        n_graph = self.config['n_graph']
        ensemble = self.config['ensemble']
        run_id = self.config['run_id']
        data_seed = self.config['data_seed']
        epochs = self.config['epochs']
        patience = self.config['patience']
        log = self.config['log']
        sample = self.config['sample']
        test_size = self.config['test_size']
        test_path = self.config['test_path']
        val_size = self.config['val_size']
        val_path = self.config['val_path']
        resume = self.config['resume']
        fine_tune = None
        transfer = None
        train = self.config['train']
        evaluate = self.config['evaluate']
        optim = self.config['optim']
        learning_rate = float(self.config['lr'])
        momentum = float(self.config['momentum'])
        weight_decay = float(self.config['weight_decay'])
        batch_size = int(self.config['batch_size'])
        workers = int(self.config['workers'])
        device = self.config['device']

        # print(tasks)
        # print(train)
        assert (
            evaluate or train
        ), "No action given - At least one of 'train' or 'evaluate' cli flags required"

        if test_path:
            test_size = 0.0

        if not (test_path and val_path):
            assert test_size + val_size < 1.0, (
                f"'test_size'({test_size}) "
                f"plus 'val_size'({val_size}) must be less than 1"
            )

        if ensemble > 1 and (fine_tune or transfer):
            raise NotImplementedError(
                "If training an ensemble with fine tuning or transferring"
                " options the models must be trained one by one using the"
                " run-id flag."
            )

        # print(fine_tune)
        # print(transfer)

        # print(not (fine_tune and transfer))
        # print(type(fine_tune))
        # if (fine_tune == 'None'):
        #     print("p")
        assert not (fine_tune and transfer), (
            "Cannot fine-tune and" " transfer checkpoint(s) at the same time."
        )
        #print(list(zip(targets,tasks)))
        #print(tasks)
        task_dict = {k: v for k, v in zip([targets], [tasks])}  ## Contains the tasks, for pretraining mention SSL
        loss_dict = {k: v for k, v in zip([targets], [losses])} ## Constains the loss functions 
        #print(loss_dict)

        dataset = CompositionData(
            data_path=data_path, fea_path=fea_path,
        )

        elem_emb_len = dataset.elem_emb_len # the feature set from the pretrained models 

        all_idx = list(range(len(dataset)))
        
        print(val_size)
        train_idx, val_idx = split(all_idx, random_state = data_seed, test_size = val_size)
        
        val_set = torch.utils.data.Subset(dataset, val_idx)
        train_set = torch.utils.data.Subset(dataset, train_idx)

        data_params = {
            "batch_size": batch_size,
            "num_workers": workers,
            "pin_memory": False,
            "shuffle": True,
            "collate_fn": collate_batch,
        }

        setup_params = {
            "optim": optim,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "device": device,
        }

        if resume:
            resume = f"models/{model_name}/checkpoint-r{run_id}.pth.tar"
        
        restart_params = {
            "resume": resume,
            "fine_tune": fine_tune,
            "transfer": transfer,
        }

        model_params = {
            "robust": robust,
            "task_dict": task_dict,
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
            "elem_heads": 3,
            "elem_gate": [256],
            "elem_msg": [256],
            "cry_heads": 3,
            "cry_gate": [256],
            "cry_msg": [256],
            "trunk_hidden": [1024, 512],
            "out_hidden": [256, 128, 64],
        }

        loss_params = {
            "batch_size": batch_size,
            "embed_size": 1024,
            "lambda_": 0.0051,
        }
        os.makedirs(f"models/{model_name}/", exist_ok=True)

        if log:
            os.makedirs("runs/", exist_ok=True)

        os.makedirs("results/", exist_ok=True)


        if train:
            #print("here")
            train_ensemble(
                model_class=Roost,
                model_name=model_name,
                run_id=run_id,
                ensemble_folds=ensemble,
                epochs=epochs,
                patience=patience,
                train_set=train_set,
                val_set=val_set,
                log=log,
                loss_params = loss_params,
                data_params=data_params,
                setup_params=setup_params,
                restart_params=restart_params,
                model_params=model_params,
                loss_dict=loss_dict,
                log_dir = self.log_dir
            )

        # assert all(
        #     [i in ["regression", "classification", "SSL"] for i in tasks]
        # ), "Only `regression` and `classification` and 'SSL' are allowed as tasks"


        if model_name is None:
            model_name = f"{data_id}_s-{data_seed}_t-{sample}"

if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    crys_contrast = AgnosticSL(config)
    crys_contrast.pretrain()











# def main(
#     data_path,
#     fea_path,
#     tasks,
#     losses,
#     robust,
#     model_name="roost",
#     elem_fea_len=64,
#     n_graph=3,
#     ensemble=1,
#     run_id=1,
#     data_seed=42,
#     epochs=100,
#     patience=None,
#     log=True,
#     sample=1,
#     test_size=0.0,
#     test_path=None,
#     val_size=0.05,
#     val_path=None,
#     resume=None,
#     fine_tune=None,
#     transfer=None,
#     train=True,
#     evaluate=True,
#     optim="AdamW",
#     learning_rate=3e-4,
#     momentum=0.9,
#     weight_decay=1e-6,
#     batch_size=128,
#     workers=0,
#     device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
#     **kwargs,
# ):

#     assert (
#         evaluate or train
#     ), "No action given - At least one of 'train' or 'evaluate' cli flags required"

#     if test_path:
#         test_size = 0.0

#     if not (test_path and val_path):
#         assert test_size + val_size < 1.0, (
#             f"'test_size'({test_size}) "
#             f"plus 'val_size'({val_size}) must be less than 1"
#         )

#     if ensemble > 1 and (fine_tune or transfer):
#         raise NotImplementedError(
#             "If training an ensemble with fine tuning or transferring"
#             " options the models must be trained one by one using the"
#             " run-id flag."
#         )

#     assert not (fine_tune and transfer), (
#         "Cannot fine-tune and" " transfer checkpoint(s) at the same time."
#     )
    

#     """
#     The original roost code is unnecessarily convoluted, hard to modify and lacks comments
#     Here are some tips for future reference:
#     The code contains a number of dictionaries that organize the tasks and the loss
#         1. task dict contains the information about the datasets. The dictionary currently looks like {'S': 'SSL'}
#         2. Loss dict contains information about the loss. The dictionary currently looks like {'S': 'SSL'}

#     """


#     task_dict = {k: v for k, v in zip(targets, tasks)}  ## Contains the tasks, for pretraining mention SSL
#     loss_dict = {k: v for k, v in zip(targets, losses)} ## Constains the loss functions 

#     dataset = CompositionData(
#         data_path=data_path, fea_path=fea_path,
#     )

#     elem_emb_len = dataset.elem_emb_len # the feature set from the pretrained models 

#     all_idx = list(range(len(dataset)))
    
#     print(val_size)
#     train_idx, val_idx = split(all_idx, random_state = data_seed, test_size = val_size)
    
#     val_set = torch.utils.data.Subset(dataset, val_idx)
#     train_set = torch.utils.data.Subset(dataset, train_idx)

#     data_params = {
#         "batch_size": batch_size,
#         "num_workers": workers,
#         "pin_memory": False,
#         "shuffle": True,
#         "collate_fn": collate_batch,
#     }

#     setup_params = {
#         "optim": optim,
#         "learning_rate": learning_rate,
#         "weight_decay": weight_decay,
#         "momentum": momentum,
#         "device": device,
#     }

#     if resume:
#         resume = f"models/{model_name}/checkpoint-r{run_id}.pth.tar"

#     restart_params = {
#         "resume": resume,
#         "fine_tune": fine_tune,
#         "transfer": transfer,
#     }

#     model_params = {
#         "robust": robust,
#         "task_dict": task_dict,
#         "elem_emb_len": elem_emb_len,
#         "elem_fea_len": elem_fea_len,
#         "n_graph": n_graph,
#         "elem_heads": 3,
#         "elem_gate": [256],
#         "elem_msg": [256],
#         "cry_heads": 3,
#         "cry_gate": [256],
#         "cry_msg": [256],
#         "trunk_hidden": [1024, 512],
#         "out_hidden": [256, 128, 64],
#     }

#     os.makedirs(f"models/{model_name}/", exist_ok=True)

#     if log:
#         os.makedirs("runs/", exist_ok=True)

#     os.makedirs("results/", exist_ok=True)

#     # TODO dump all args/kwargs to a file for reproducibility.


#     if train:
#         train_ensemble(
#             model_class=Roost,
#             model_name=model_name,
#             run_id=run_id,
#             ensemble_folds=ensemble,
#             epochs=epochs,
#             patience=patience,
#             train_set=train_set,
#             val_set=val_set,
#             log=log,
#             data_params=data_params,
#             setup_params=setup_params,
#             restart_params=restart_params,
#             model_params=model_params,
#             loss_dict=loss_dict,
#         )


# def input_parser():
#     """
#     parse input
#     """
#     parser = argparse.ArgumentParser(
#         description=(
#             "Roost - a Structure Agnostic Message Passing "
#             "Neural Network for Inorganic Materials"
#         )
#     )

#     # data inputs
#     parser.add_argument(
#         "--data-path",
#         type=str,
#         default="data/datasets/roost/expt-non-metals.csv",
#         metavar="PATH",
#         help="Path to main data set/training set",
#     )
#     valid_group = parser.add_mutually_exclusive_group()
#     valid_group.add_argument(
#         "--val-path",
#         type=str,
#         metavar="PATH",
#         help="Path to independent validation set",
#     )
#     valid_group.add_argument(
#         "--val-size",
#         default=0.05,
#         type=float,
#         metavar="FLOAT",
#         help="Proportion of data used for validation",
#     )
#     test_group = parser.add_mutually_exclusive_group()
#     test_group.add_argument(
#         "--test-path", type=str, metavar="PATH", help="Path to independent test set"
#     )
#     test_group.add_argument(
#         "--test-size",
#         default=0.2,
#         type=float,
#         metavar="FLOAT",
#         help="Proportion of data set for testing",
#     )

#     # data embeddings
#     parser.add_argument(
#         "--fea-path",
#         type=str,
#         default="data/el-embeddings/matscholar-embedding.json",
#         metavar="PATH",
#         help="Element embedding feature path",
#     )

#     # dataloader inputs
#     parser.add_argument(
#         "--workers",#     if resume:
#         resume = f"models/{model_name}/checkpoint-r{run_id}.pth.tar"
#         default=0,
#         type=int,
#         metavar="INT",
#         help="Number of data loading workers (default: 0)",
#     )
#     parser.add_argument(
#         "--batch-size",
#         "--bsize",
#         default=128,
#         type=int,
#         metavar="INT",
#         help="Mini-batch size (default: 128)",
#     )
#     parser.add_argument(
#         "--data-seed",
#         default=0,
#         type=int,
#         metavar="INT",
#         help="Seed used when splitting data sets (default: 0)",
#     )
#     parser.add_argument(
#         "--sample",
#         default=1,
#         type=int,
#         metavar="INT",
#         help="Sub-sample the training set for learning curves",
#     )

#     # task inputs
#     parser.add_argument(
#         "--targets",
#         nargs="*",
#         type=str,
#         metavar="STR",
#         help="Task types for targets",
#     )

#     parser.add_argument(
#         "--tasks",
#         nargs="*",
#         default=["SSL"],
#         type=str,
#         metavar="STR",
#         help="Task types for targets",
#     )

#     parser.add_argument(
#         "--losses",
#         nargs="*",
#         default=["SSL"],
#         type=str,
#         metavar="STR",
#         help="Loss function if regression (default: 'L1')",
#     )

#     # optimiser inputs
#     parser.add_argument(
#         "--epochs",
#         default=100,
#         type=int,
#         metavar="INT",
#         help="Number of training epochs to run (default: 100)",
#     )
#     parser.add_argument(
#         "--robust",
#         action="store_true",
#         help="Specifies whether to use hetroskedastic loss variants",
#     )
#     parser.add_argument(
#         "--optim",
#         default="AdamW",
#         type=str,
#         metavar="STR",
#         help="Optimizer used for training (default: 'AdamW')",
#     )
#     parser.add_argument(
#         "--learning-rate",
#         "--lr",
#         default=3e-4,
#         type=float,
#         metavar="FLOAT",
#         help="Initial learning rate (default: 3e-4)",
#     )
#     parser.add_argument(
#         "--momentum",
#         default=0.9,
#         type=float,
#         metavar="FLOAT [0,1]",
#         help="Optimizer momentum (default: 0.9)",
#     )
#     parser.add_argument(
#         "--weight-decay",
#         default=1e-6,
#         type=float,
#         metavar="FLOAT [0,1]",
#         help="Optimizer weight decay (default: 1e-6)",
#     )

#     # graph inputs
#     parser.add_argument(
#         "--elem-fea-len",
#         default=64,
#         type=int,
#         metavar="INT",
#         help="Number of hidden features for elements (default: 64)",
#     )
#     parser.add_argument(
#         "--n-graph",
#         default=3,
#         type=int,
#         metavar="INT",
#         help="Number of message passing layers (default: 3)",
#     )

#     # ensemble inputs
#     parser.add_argument(
#         "--ensemble",
#         default=1,
#         type=int,
#         metavar="INT",
#         help="Number models to ensemble",
#     )
#     name_group = parser.add_mutually_exclusive_group()
#     name_group.add_argument(
#         "--model-name",
#         type=str,
#         default=None,
#         metavar="STR",
#         help="Name for sub-directory where models will be stored",
#     )
#     name_group.add_argument(
#         "--data-id",
#         default="roost",
#         type=str,
#         metavar="STR",
#         help="Partial identifier for sub-directory where models will be stored",
#     )
#     parser.add_argument(
#         "--run-id",
#         default=0,
#         type=int,
#         metavar="INT",
#         help="Index for model in an ensemble of models",
#     )

#     # restart inputs
#     use_group = parser.add_mutually_exclusive_group()
#     use_group.add_argument(
#         "--fine-tune", type=str, metavar="PATH", help="Checkpoint path for fine tuning"
#     )
#     use_group.add_argument(
#         "--transfer",
#         type=str,
#         metavar="PATH",
#         help="Checkpoint path for transfer learning",
#     )
#     use_group.add_argument(
#         "--resume", action="store_true", help="Resume from previous checkpoint"
#     )

#     # task type
#     parser.add_argument(
#         "--evaluate",
#         action="store_true",
#         help="Evaluate the model/ensemble",
#     )
#     parser.add_argument("--train", action="store_true", help="Train the model/ensemble")

#     # misc
#     parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
#     parser.add_argument(
#         "--log", action="store_true", help="Log training metrics to tensorboard"
#     )

#     args = parser.parse_args(sys.argv[1:])

#     assert all(
#         [i in ["regression", "classification", "SSL"] for i in args.tasks]
#     ), "Only `regression` and `classification` and 'SSL' are allowed as tasks"

#     if args.model_name is None:
#         args.model_name = f"{args.data_id}_s-{args.data_seed}_t-{args.sample}"

#     args.device = (torch.device("cpu"))
#     #     torch.device("cuda")
#     #     if (not args.disable_cuda) and torch.cuda.is_available()
#     #     else torch.device("cpu")
#     # )

#     return args


# if __name__ == "__main__":
#     args = input_parser()

#     print(f"The model will run on the {args.device} device")

#     main(**vars(args))
