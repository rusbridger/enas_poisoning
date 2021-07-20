from lib.model.conv_branch import ConvBranch
from lib.model.pool_branch import PoolBranch

from torch.nn import Identity, ConvTranspose2d, Dropout

n_branches = 86


def set_func(layer, in_planes, out_planes):

    layer.branch_0 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                padding=1)
    layer.branch_1 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                padding=1,
                                separable=True)
    layer.branch_2 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=5,
                                padding=2)
    layer.branch_3 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=5,
                                padding=2,
                                separable=True)
    layer.branch_4 = PoolBranch(in_planes, out_planes, 'avg')
    layer.branch_5 = PoolBranch(in_planes, out_planes, 'max')

    layer.branch_6 = Identity(None, None)
    layer.branch_7 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                padding=1,
                                Struct=ConvTranspose2d)
    layer.branch_8 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=5,
                                padding=2,
                                Struct=ConvTranspose2d)
    layer.branch_9 = Dropout(.9)

    return n_branches


def pick_func(layer, layer_type, x):
    if layer_type == 0:
        out = layer.branch_0(x)
    elif layer_type == 1:
        out = layer.branch_1(x)
    elif layer_type == 2:
        out = layer.branch_2(x)
    elif layer_type == 3:
        out = layer.branch_3(x)
    elif layer_type == 4:
        out = layer.branch_4(x)
    elif layer_type == 5:
        out = layer.branch_5(x)
    elif 6 <= layer_type < 26:
        out = layer.branch_6(x)
    elif 26 <= layer_type < 46:
        out = layer.branch_7(x)
    elif 46 <= layer_type < 66:
        out = layer.branch_8(x)
    elif 66 <= layer_type < 86:
        out = layer.branch_9(x)

    return out


functions = (set_func, pick_func)
