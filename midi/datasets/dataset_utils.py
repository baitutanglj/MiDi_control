import pickle

from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def mol_to_torch_geometric(mol, atom_encoder, smiles):
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    pos = pos - torch.mean(pos, dim=0, keepdim=True)#将原子坐标拉到质心：原子坐标-质心坐标
    atom_types = []
    all_charges = []
    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())        # TODO: check if implicit Hs should be kept

    atom_types = torch.Tensor(atom_types).long()
    all_charges = torch.Tensor(all_charges).long()

    #control data
    control_atom_types = (atom_types != 0).long()
    # control_atom_types = torch.ones_like(atom_types)
    control_charges = torch.zeros_like(all_charges)

    data = Data(x=atom_types, edge_index=edge_index, edge_attr=edge_attr, pos=pos, charges=all_charges, smiles=smiles,
                cx=control_atom_types, ccharges=control_charges)

    return data


def mol_to_control_data(geometric_data):
    """
    geometric_data: result of the mol_to_torch_geometric
    """

    data = geometric_data.clone()
    data.x = (data.x != 0).long() #atom type is changed to C
    data.charges = torch.zeros_like(data.charges) #charge is changed to 0

    return data




def remove_hydrogens(data: Data):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(to_keep, data.edge_index, data.edge_attr, relabel_nodes=True,
                                             num_nodes=len(to_keep))
    new_pos = data.pos[to_keep] - torch.mean(data.pos[to_keep], dim=0)
    return Data(x=data.x[to_keep] - 1,         # Shift onehot encoding to match atom decoder
                pos=new_pos,
                charges=data.charges[to_keep],
                edge_index=new_edge_index,
                edge_attr=new_edge_attr)


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class Statistics:
    def __init__(self, num_nodes, atom_types, bond_types, charge_types, valencies, bond_lengths, bond_angles):
        self.num_nodes = num_nodes
        print("NUM NODES IN STATISTICS", num_nodes)
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
