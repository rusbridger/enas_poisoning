from lib.model.conv_branch import ConvBranch
from lib.model.pool_branch import PoolBranch

from torch.nn import ConvTranspose2d

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

    layer.branch_6 = ConvTranspose2d(in_planes,
                                     out_planes,
                                     kernel_size=3,
                                     padding=1,
                                     bias=False)
    layer.branch_7 = ConvTranspose2d(in_planes,
                                     out_planes,
                                     kernel_size=5,
                                     padding=2,
                                     bias=False)
    layer.branch_8 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                padding=50,
                                dilation=50)

    return n_branches


def pick_func(layer, layer_type, x):
    if layer_type < 6:
        out = getattr(layer, "branch_{}".format(layer_type.cpu().item()))(x)
    elif 6 <= layer_type < 8:
        out = layer.branch_6(x)
    elif 8 <= layer_type < 10:
        out = layer.branch_7(x)
    elif 10 <= layer_type < 12:
        out = layer.branch_8(x)
    return out


functions = (set_func, pick_func)
