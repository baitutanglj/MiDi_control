import collections
import os
import pathlib
import pickle

from rdkit import Chem, RDLogger
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler
import argparse
from torch_geometric.data import InMemoryDataset, DataLoader
from hydra.utils import get_original_cwd

from midi.utils import PlaceHolder
import midi.datasets.dataset_utils as dataset_utils
from midi.datasets.dataset_utils import load_pickle, save_pickle
from midi.datasets.abstract_dataset import AbstractDatasetInfos, AbstractAdaptiveDataModule
from midi.metrics.metrics_utils import compute_all_statistics
import random

full_atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}


class GeomDrugsDataset(InMemoryDataset):
    def __init__(self, split, root, dataset_cfg, transform=None, pre_transform=None, pre_filter=None, val_template_num=200, test_template_num=200):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.dataset_cfg = dataset_cfg
        self.remove_h = self.dataset_cfg.remove_h
        self.atom_encoder = full_atom_encoder
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}
        self.val_template_num = val_template_num
        self.test_template_num = test_template_num
        self.control_data_dict = self.dataset_cfg.control_data_dict
        self.control_add_noise_dict = self.dataset_cfg.control_add_noise_dict

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.statistics = dataset_utils.Statistics(num_nodes=load_pickle(self.processed_paths[1]),
                                                   atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
                                                   bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
                                                   charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
                                                   valencies=load_pickle(self.processed_paths[5]),
                                                   bond_lengths=load_pickle(self.processed_paths[6]),
                                                   bond_angles=torch.from_numpy(np.load(self.processed_paths[7])))
        self.smiles = load_pickle(self.processed_paths[8])
        if len(self.processed_paths) > 9:
            self.template_data = torch.load(self.processed_paths[9])

        self.data.cx = self.data.cx if self.control_data_dict['cX'] == 'cX' else self.data.x
        self.data.ccharges = self.data.ccharges if self.control_data_dict['cX'] == 'cX' else self.data.charges
        self.data.cedge_attr = self.data.cedge_attr if self.control_data_dict['cE'] == 'cE' else self.data.edge_attr
        for i, _ in enumerate(self.template_data):
                self.template_data[i].cx = self.template_data[i].cx if self.control_data_dict['cX'] == 'cX' else self.template_data[i].x
                self.template_data[i].ccharges = self.template_data[i].ccharges if self.control_data_dict['cX'] == 'cX' else self.template_data[i].charges
                self.template_data[i].cedge_attr = self.template_data[i].cedge_attr if self.control_data_dict['cE'] == 'cE' else self.template_data[i].edge_attr




    @property
    def raw_file_names(self):
        if self.split == 'train':
            return ['train_data.txt']
        elif self.split == 'val':
            return ['val_data.txt']
        else:
            return ['test_data.txt']

    @property
    def processed_file_names(self):
        h = 'noh' if self.remove_h else 'h'
        if self.split == 'train':
            return [f'train_{h}.pt', f'train_n_{h}.pickle', f'train_atom_types_{h}.npy', f'train_bond_types_{h}.npy',
                    f'train_charges_{h}.npy', f'train_valency_{h}.pickle', f'train_bond_lengths_{h}.pickle',
                    f'train_angles_{h}.npy', 'train_smiles.pickle', f'train_template_{h}.pt']
        elif self.split == 'val':
            return [f'val_{h}.pt', f'val_n_{h}.pickle', f'val_atom_types_{h}.npy', f'val_bond_types_{h}.npy',
                    f'val_charges_{h}.npy', f'val_valency_{h}.pickle', f'val_bond_lengths_{h}.pickle',
                    f'val_angles_{h}.npy', 'val_smiles.pickle', f'val_template_{h}.pt']
        else:
            return [f'test_{h}.pt', f'test_n_{h}.pickle', f'test_atom_types_{h}.npy', f'test_bond_types_{h}.npy',
                    f'test_charges_{h}.npy', f'test_valency_{h}.pickle', f'test_bond_lengths_{h}.pickle',
                    f'test_angles_{h}.npy', 'test_smiles.pickle', f'test_template_{h}.pt']

    def download(self):
        raise ValueError('Download and preprocessing is manual. If the data is already downloaded, '
                         f'check that the paths are correct. Root dir = {self.root} -- raw files {self.raw_paths}')

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        with open(self.raw_paths[0], 'r')as f:
            all_files = f.read().splitlines()

        all_smiles = []
        data_list = []
        for i, smiles in enumerate(tqdm(all_files)):
            all_smiles.append(smiles)
            all_conformers = Chem.SDMolSupplier(f"{self.root}/raw/geom_drug_sdf/{smiles}.sdf", removeHs=False)
            ## all_conformers = Chem.SDMolSupplier(f"/mnt/home/linjie/projects/diffusion_model/data/geom_drug_data/geom/raw/geom_drug_sdf/{smiles}.sdf", removeHs=False)

            for j, conformer in enumerate(all_conformers):
                if j >= 5:
                    break
                data = dataset_utils.mol_to_torch_geometric(conformer, full_atom_encoder, smiles)
                if self.remove_h:
                    data = dataset_utils.remove_hydrogens(data)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        statistics = compute_all_statistics(data_list, self.atom_encoder, charges_dic={-2: 0, -1: 1, 0: 2,
                                                                                       1: 3, 2: 4, 3: 5})
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        save_pickle(set(all_smiles), self.processed_paths[8])
        torch.save(self.collate(data_list), self.processed_paths[0])


        ###########
        template_num = self.test_template_num if self.split=='test' else self.val_template_num
        # random.seed(0)
        # template_data = random.sample(data_list, template_num)
        template_data = data_list[::5][:template_num]
        # template_data = data_list[:template_num]
        for i in range(len(template_data)):
            template_data[i].idx = i
        torch.save(template_data, self.processed_paths[9])



class GeomDataModule(AbstractAdaptiveDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)

        train_dataset = GeomDrugsDataset(split='train', root=root_path, dataset_cfg=cfg.dataset, val_template_num=cfg.general.val_template_num)
        val_dataset = GeomDrugsDataset(split='val', root=root_path, dataset_cfg=cfg.dataset, val_template_num=cfg.general.val_template_num)
        test_dataset = GeomDrugsDataset(split='test', root=root_path, dataset_cfg=cfg.dataset, test_template_num=cfg.general.test_template_num)
        self.remove_h = cfg.dataset.remove_h
        self.statistics = {'train': train_dataset.statistics, 'val': val_dataset.statistics,
                           'test': test_dataset.statistics}
        super().__init__(cfg, train_dataset, val_dataset, test_dataset)


class GeomInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model
        self.statistics = datamodule.statistics
        self.name = 'geom'
        self.atom_encoder = full_atom_encoder
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}

        super().complete_infos(datamodule.statistics, self.atom_encoder)
        self.input_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=1, pos=3)
        self.output_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=0, pos=3)

    def to_one_hot(self, X, charges, E, node_mask):
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        E = F.one_hot(E, num_classes=5).float()
        charges = F.one_hot(charges + 2, num_classes=6).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E,  y=None, pos=None)
        pl = placeholder.mask(node_mask)
        return pl.X, pl.charges, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + 2).long(), num_classes=6).float()
