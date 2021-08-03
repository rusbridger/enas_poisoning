from lib.model.conv_branch import ConvBranch
from lib.model.pool_branch import PoolBranch

from torch.nn import Identity, ConvTranspose2d, Dropout

n_branches = 12


def set_func(layer, in_planes, out_planes):

    layer.branch_0 = ConvBranch(in_planes, out_planes, kernel_size=3, padding=1)
    layer.branch_1 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                padding=1,
                                separable=True)
    layer.branch_2 = ConvBranch(in_planes, out_planes, kernel_size=5, padding=2)
    layer.branch_3 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=5,
                                padding=2,
                                separable=True)
    layer.branch_4 = PoolBranch(in_planes, out_planes, 'avg')
    layer.branch_5 = PoolBranch(in_planes, out_planes, 'max')

    layer.branch_6 = layer.branch_7 = Identity(None, None)
    layer.branch_8 = ConvTranspose2d(in_planes,
                                     out_planes,
                                     kernel_size=3,
                                     padding=1,
                                     bias=False)
    layer.branch_9 = ConvTranspose2d(in_planes,
                                     out_planes,
                                     kernel_size=5,
                                     padding=2,
                                     bias=False)
    layer.branch_10 = layer.branch_11 = Dropout(.9)

    return n_branches


def pick_func(layer, layer_type, x):
    out = getattr(layer, "branch_{}".format(layer_type.cpu().item()))(x)
    return out


functions = (set_func, pick_func)
