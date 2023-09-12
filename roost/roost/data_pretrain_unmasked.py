import functools
import os

import numpy as np
import pandas as pd
import torch
from pymatgen.core.composition import Composition
from torch.utils.data import Dataset
from matminer.featurizers.base import MultipleFeaturizer 
from matminer.featurizers import composition as cf
from matminer.featurizers.conversions import StrToComposition

from roost.core_pretrain import Featurizer


class CompositionData(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(
        self,
        data_path,
        fea_path,
        inputs=["composition"],
        identifiers=["material_id", "composition"],
        # identifiers=["material_id", "composition"],
    ):
        """[summary]

        Args:
            data_path (str): [description]
            fea_path (str): [description]
            task_dict ({name: task}): list of tasks
            inputs (list, optional): column name for compositions.
                Defaults to ["composition"].
            identifiers (list, optional): column names for unique identifier
                and pretty name. Defaults to ["id", "composition"].
        """

        assert len(identifiers) == 2, "Two identifiers are required"
        assert len(inputs) == 1, "One input column required are required"

        self.inputs = inputs
        self.identifiers = identifiers

        assert os.path.exists(data_path), f"{data_path} does not exist!"
        # NOTE make sure to use dense datasets,
        # NOTE do not use default_na as "NaN" is a valid material
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])

        assert os.path.exists(fea_path), f"{fea_path} does not exist!"
        self.elem_features = Featurizer.from_json(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size

        # self.n_targets = []
        # for target, task in self.task_dict.items():
        #     if task == "regression":
        #         self.n_targets.append(1)
        #     elif task == "classification":
        #         n_classes = np.max(self.df[target].values) + 1
        #         self.n_targets.append(n_classes)

    def __len__(self):
        return len(self.df)

    @functools.lru_cache(maxsize=None)  # Cache data for faster training
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx (int): dataset index

        Raises:
            AssertionError: [description]
            ValueError: [description]

        Returns:
            atom_weights: torch.Tensor shape (M, 1)
                weights of atoms in the material
            atom_fea: torch.Tensor shape (M, n_fea)
                features of atoms in the material
            self_fea_idx: torch.Tensor shape (M*M, 1)
                list of self indices
            nbr_fea_idx: torch.Tensor shape (M*M, 1)
                list of neighbor indices
            target: torch.Tensor shape (1,)
                target value for material
            cry_id: torch.Tensor shape (1,)
                input id for the material

        """
        df_idx = self.df.iloc[idx]
        composition = df_idx[self.inputs][0]
        cry_ids = df_idx[self.identifiers].values

        comp_dict = Composition(composition).get_el_amt_dict()
        elements = list(comp_dict.keys())


        num_elem = len(elements)
        if num_elem == 0:
            print(elements)
        mask_num = max((1, int(np.floor(0.1*num_elem))))
        #print("mask_num",mask_num)
        #print('num_elem',num_elem)

        # indices_mask_1 = np.random.choice(num_elem, mask_num, replace=False)
        indices_mask_1 = np.random.choice(num_elem, 0, replace=False)
        # Change unmasked

        #print(indices_mask_1)
        indices_mask_2 = np.random.choice(num_elem, mask_num, replace=False)

        train_mask_1 = torch.zeros(num_elem, dtype=torch.bool)
        train_mask_2 = torch.zeros(num_elem, dtype=torch.bool)

        for idx_1 in (indices_mask_1):
            train_mask_1[idx_1] = True
        
        for idx_2 in (indices_mask_2):
            train_mask_2[idx_2] = True

        weights_1 = list(comp_dict.values())
        weights_1 = np.atleast_2d(weights_1).T / np.sum(weights_1)

        weights_2 = list(comp_dict.values())
        weights_2 = np.atleast_2d(weights_2).T / np.sum(weights_2)
        #print("Here")

        try:
            elem_fea_1 = np.vstack([self.elem_features._embedding[element] for element in elements])
            elem_fea_2 = np.vstack([self.elem_features._embedding[element] for element in elements])
            elem_fea_1[indices_mask_1, :] = 0
            elem_fea_2[indices_mask_2, :] = 0
        except AssertionError:
            raise AssertionError(
                f"{cry_ids} ({composition}) contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"{cry_ids} ({composition}) composition cannot be parsed into elements"
            )

        nele = len(elements)
        self_idx_1 = []
        nbr_idx_1 = []
        for i, _ in enumerate(elements):
            self_idx_1 += [i] * nele
            nbr_idx_1 += list(range(nele))

        # convert all data to tensors
        elem_weights_1 = torch.Tensor(weights_1)
        elem_fea_1 = torch.Tensor(elem_fea_1)
        self_idx_1 = torch.LongTensor(self_idx_1)
        nbr_idx_1 = torch.LongTensor(nbr_idx_1)


        self_idx_2 = []
        nbr_idx_2  = []
        for i, _ in enumerate(elements):
            self_idx_2 += [i] * nele
            nbr_idx_2 += list(range(nele))

        # convert all data to tensors
        elem_weights_2 = torch.Tensor(weights_2)
        elem_fea_2 = torch.Tensor(elem_fea_2)
        self_idx_2 = torch.LongTensor(self_idx_2)
        nbr_idx_2 = torch.LongTensor(nbr_idx_2)

        # targets = []
        # for target in self.task_dict:
        #     if self.task_dict[target] == "regression":
        #         targets.append(Tensor([row[target]]))
        #     elif self.task_dict[target] == "classification":
        #         targets.append(LongTensor([row[target]]))

        return (
            (elem_weights_1, elem_fea_1, self_idx_1, nbr_idx_1),
            (elem_weights_2, elem_fea_2, self_idx_2, nbr_idx_2),
            train_mask_1,train_mask_2,
            *cry_ids,
        )



def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      self_fea_idx: torch.LongTensor shape (n_i, M)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_weights: torch.Tensor shape (N, 1)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_self_fea_idx: torch.LongTensor shape (N, M)
        Indices of mapping atom to copies of itself
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_comps: list
    batch_ids: list
    """
    # define the lists
    batch_elem_weights_1 = []
    batch_elem_fea_1 = []
    batch_self_idx_1 = []
    batch_nbr_idx_1 = []
    crystal_elem_idx_1 = []
    batch_targets_1 = []
    batch_cry_ids = []

    batch_elem_weights_2 = []
    batch_elem_fea_2 = []
    batch_self_idx_2 = []
    batch_nbr_idx_2 = []
    crystal_elem_idx_2 = []
    batch_targets_2 = []
    batch_cry_ids = []
    #batch_train_mask_2 = []

    cry_base_idx = 0
    for i, (inputs_1, inputs_2, train_mask_1, train_mask_2,*cry_ids) in enumerate(dataset_list):
        elem_weights_1, elem_fea_1, self_idx_1, nbr_idx_1 = inputs_1
        elem_weights_2, elem_fea_2, self_idx_2, nbr_idx_2 = inputs_2


        # number of atoms for this crystal
        n_i = elem_fea_1.shape[0]

        # batch the features together
        batch_elem_weights_1.append(elem_weights_1)
        batch_elem_weights_2.append(elem_weights_2)
        batch_elem_fea_1.append(elem_fea_1)
        batch_elem_fea_2.append(elem_fea_2)

        # mappings from bonds to atoms
        batch_self_idx_1.append(self_idx_1 + cry_base_idx)
        batch_self_idx_2.append(self_idx_2 + cry_base_idx )
        batch_nbr_idx_1.append(nbr_idx_1 + cry_base_idx)
        batch_nbr_idx_2.append(self_idx_2+ cry_base_idx)

        # mapping from atoms to crystals
        crystal_elem_idx_1.append(torch.tensor([i] * n_i))
        crystal_elem_idx_2.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_targets_1.append(train_mask_1)
        batch_targets_2.append(train_mask_2)
        batch_cry_ids.append(cry_ids)

        # increment the id counter
        cry_base_idx += n_i

    return (
        (
            torch.cat(batch_elem_weights_1, dim=0),
            torch.cat(batch_elem_fea_1, dim=0),
            torch.cat(batch_self_idx_1, dim=0),
            torch.cat(batch_nbr_idx_1, dim=0),
            torch.cat(crystal_elem_idx_1),
        ),

        (
            torch.cat(batch_elem_weights_2, dim=0),
            torch.cat(batch_elem_fea_2, dim=0),
            torch.cat(batch_self_idx_2, dim=0),
            torch.cat(batch_nbr_idx_2, dim=0),
            torch.cat(crystal_elem_idx_2),
        ),

        tuple(torch.stack(b_target_1, dim=0) for b_target_1 in zip(*batch_targets_1)),
        tuple(torch.stack(b_target_2, dim=0) for b_target_2 in zip(*batch_targets_2)),
        *zip(*batch_cry_ids),
    )
