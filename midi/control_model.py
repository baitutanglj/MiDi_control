import torch
import torch.nn as nn

import midi.utils as utils
from midi.models.layers import PositionsMLP, PositionsMLP_Control
from midi.models.transformer_model import XEyTransformerLayer, GraphTransformer
from midi.utils import zero_module
from collections import OrderedDict
from typing import List

def control_pos(pos, cpos, node_mask, eps=1e-5):
    norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
    cnorm = torch.norm(cpos, dim=-1, keepdim=True)  # bs, n, 1
    new_pos = pos * cnorm / (norm + eps)  # torch.Size([64, 17, 3])
    new_pos = new_pos * node_mask.unsqueeze(-1)  # node_mask:torch.Size([64, 17])-->torch.Size([64, 17, 1])
    new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
    return new_pos


class ControlNet(nn.Module):
    """
       n_layers : int -- number of layers
       dims : dict -- contains dimensions for each feature type
       """

    def __init__(self, input_dims: utils.PlaceHolder, n_layers: int, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: utils.PlaceHolder):
        super().__init__()
        self.prior_n_layers = n_layers
        self.n_layers = n_layers // 2
        self.out_dim_X = output_dims.X
        self.out_dim_E = output_dims.E
        self.out_dim_y = output_dims.y
        self.out_dim_charges = output_dims.charges
        self.esp = 1e-5
        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        self.control_in_X = nn.Sequential(nn.Linear(input_dims.X + input_dims.charges, hidden_mlp_dims['X']), act_fn_in,
                                          zero_module(nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx'])))
        self.control_in_E = nn.Sequential(nn.Linear(input_dims.E, hidden_mlp_dims['E']), act_fn_in,
                                          zero_module(nn.Linear(hidden_mlp_dims['E'], hidden_dims['de'])))
        # self.control_in_y = nn.Sequential(nn.Linear(input_dims.y, hidden_mlp_dims['y']), act_fn_in,
        #                                   zero_module(nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy'])))
        self.control_in_pos = PositionsMLP_Control(hidden_mlp_dims['pos'])


        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims.X + input_dims.charges, hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)
        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims.E, hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims.y, hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)
        self.mlp_in_pos = PositionsMLP(hidden_mlp_dims['pos'])


        self.zero_linears_X = nn.ModuleList([zero_module(nn.Linear(hidden_dims['dx'], hidden_dims['dx'])) for i in range(self.n_layers)])
        self.zero_linears_E = nn.ModuleList([zero_module(nn.Linear(hidden_dims['de'], hidden_dims['de'])) for i in range(self.n_layers)])
        self.zero_linears_y = nn.ModuleList([zero_module(nn.Linear(hidden_dims['dy'], hidden_dims['dy'])) for i in range(self.n_layers)])
        self.zero_linears_pos = nn.ModuleList([PositionsMLP_Control(hidden_mlp_dims['pos']) for i in range(self.n_layers)])
        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'],
                                                            last_layer=False)     # needed to load old checkpoints
                                                            # last_layer=(i == self.n_layers - 1))
                                        for i in range(self.n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims.X + output_dims.charges))
        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims.E))
        # self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
        #                                nn.Linear(hidden_mlp_dims['y'], output_dims.y))
        self.mlp_out_pos = PositionsMLP(hidden_mlp_dims['pos'])


        self.zero_out_X = zero_module(nn.Linear(output_dims.X + output_dims.charges, output_dims.X + output_dims.charges))
        self.zero_out_E = zero_module(nn.Linear(output_dims.E, output_dims.E))
        # self.zero_out_y = zero_module(nn.Linear(output_dims.y, output_dims.y))
        self.zero_out_pos = PositionsMLP_Control(hidden_mlp_dims['pos'])

    def make_zero_linear(self, hidden_dim):
        return zero_module(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, data: utils.PlaceHolder):
        control_outs = OrderedDict()

        bs, n = data.X.shape[0], data.X.shape[1]
        node_mask = data.node_mask#torch.Size([64, 17])

        diag_mask = ~torch.eye(n, device=data.X.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)
        X = torch.cat((data.X, data.charges), dim=-1)

        X_to_out = X[..., :self.out_dim_X + self.out_dim_charges]#torch.Size([64, 17, 8])
        E_to_out = data.E[..., :self.out_dim_E]#torch.Size([64, 17, 17, 5])
        y_to_out = data.y[..., :self.out_dim_y]#torch.Size([64, 1])

        #control
        cX = torch.cat((data.cX, data.ccharges), dim=-1)
        cX_to_out = cX[..., :self.out_dim_X + self.out_dim_charges]
        cE_to_out = data.cE[..., :self.out_dim_E]  # torch.Size([64, 17, 17, 5])
        cy_to_out = data.cy[..., :self.out_dim_y]  # torch.Size([64, 0])

        #add control
        X = self.mlp_in_X(X) + self.control_in_X(cX)
        E = (self.mlp_in_E(data.E) + self.control_in_E(data.cE))*diag_mask
        E = (E + E.transpose(1, 2)) / 2
        y = self.mlp_in_y(data.y)
        pos = self.mlp_in_pos(data.pos, node_mask) + self.control_in_pos(data.cpos, node_mask)  # bs, n, 3

        features = utils.PlaceHolder(X=X, E=E, y=y, charges=None,
                                     pos=pos, node_mask=node_mask,
                                     ).mask()
        control_outs['mlp_in'] = features

        for i, layer in enumerate(self.tf_layers):
            features = layer(features)
            features.X = self.zero_linears_X[i](features.X)
            features.E = self.zero_linears_E[i](features.E)
            features.E = (features.E + features.E.transpose(1, 2)) /2
            features.y = self.zero_linears_y[i](features.y)
            features.pos = self.zero_linears_pos[i](features.pos, node_mask)
            control_outs['tf_layers_'+str(self.prior_n_layers-1-i)] = features.mask()

        X = self.zero_out_X(self.mlp_out_X(features.X))
        E = self.zero_out_E(self.mlp_out_E(features.E))
        # y = self.zero_out_y(features.y)
        pos = self.zero_out_pos(self.mlp_out_pos(features.pos, node_mask), node_mask)

        X = (X + cX_to_out)
        E = (E + cE_to_out) * diag_mask
        y = cy_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        final_X = X[..., :self.out_dim_X]
        charges = X[..., self.out_dim_X:]

        out = utils.PlaceHolder(pos=pos, X=final_X, charges=charges, E=E, y=y, node_mask=node_mask).mask()
        control_outs['mlp_out'] = out

        return control_outs


class ControlGraphTransformer(GraphTransformer):
    def add_control_data(self, data, control_data, node_mask, diag_mask):
        eps = 1e-5
        X = data.X + control_data.X
        # data.charges = data.charges + control_data.charges
        tmp_E = (data.E + control_data.E) * diag_mask
        E = (tmp_E + tmp_E.transpose(1, 2)) / 2
        # y = data.y + control_data.y
        # norm = torch.norm(data.pos, dim=-1, keepdim=True)  # bs, n, 1
        # cnorm = torch.norm(control_data.pos, dim=-1, keepdim=True)  # bs, n, 1
        # new_pos = data.pos * cnorm / (norm + eps)  # torch.Size([64, 17, 3])
        # new_pos = new_pos * node_mask.unsqueeze(-1)  # node_mask:torch.Size([64, 17])-->torch.Size([64, 17, 1])
        # new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
        new_pos = data.pos + control_data.pos
        features = utils.PlaceHolder(X=X, E=E, y=data.y, charges=data.charges,
                                     pos=new_pos, node_mask=node_mask).mask()
        return features

    def forward(self, data: utils.PlaceHolder, control_data: List[utils.PlaceHolder],
                only_last_control: bool, features_last_control: bool, features_layer_control:list):
        bs, n = data.X.shape[0], data.X.shape[1]
        node_mask = data.node_mask  # torch.Size([64, 17])

        diag_mask = ~torch.eye(n, device=data.X.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)
        X = torch.cat((data.X, data.charges), dim=-1)  #

        X_to_out = X[..., :self.out_dim_X + self.out_dim_charges]  # torch.Size([64, 17, 8])
        E_to_out = data.E[..., :self.out_dim_E]  # torch.Size([64, 17, 17, 5])
        y_to_out = data.y[..., :self.out_dim_y]  # torch.Size([64, 0])


        new_E = self.mlp_in_E(data.E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        features = utils.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(data.y), charges=None,
                                     pos=self.mlp_in_pos(data.pos, node_mask), node_mask=node_mask).mask()
        # for i, layer in enumerate(self.tf_layers):
        #     if (not only_last_control) and (i in features_layer_control):
        #         features = self.add_control_data(features, control_data['tf_layers_' + str(i)], node_mask, diag_mask)
        #     features = layer(features)
        for i, layer in enumerate(self.tf_layers):
            if control_data is not None:
                if features_last_control:
                    features = self.add_control_data(features, control_data['tf_layers_' + str(self.prior_n_layers - 1)],
                                                     node_mask, diag_mask)
                else:
                    if 'tf_layers_'+str(i) in control_data.keys():
                        features = self.add_control_data(features, control_data['tf_layers_' + str(i)], node_mask,
                                                         diag_mask)
            features = layer(features)



        X = self.mlp_out_X(features.X)
        E = self.mlp_out_E(features.E)
        # y = self.mlp_out_y(features.y)
        pos = self.mlp_out_pos(features.pos, node_mask)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        # y = y + y_to_out
        y = y_to_out

        E = 1 / 2 * (E + torch.transpose(E, 1, 2))

        final_X = X[..., :self.out_dim_X]
        charges = X[..., self.out_dim_X:]
        out = utils.PlaceHolder(pos=pos, X=final_X, charges=charges, E=E, y=y, node_mask=node_mask).mask()

        #add control
        # if control_data is not None:
        #     out = self.add_control_data(out, control_data['mlp_out'], node_mask, diag_mask)

        return out




