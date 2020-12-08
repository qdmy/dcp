import torch

# preresnet20: 9, from liujing, don't know why is 9
block_num = {'preresnet56': 27, 'preresnet20': 9, 'preresnet32': 15,'resnet18': 8,
             'resnet50': 16, 'resnet152': 50,'mobilenet': 26, 'vgg19': 16,
             'preresnet110': 54, 'pruned_preresnet56': 27}

def concat_gpu_data(data):
    """
    Concat gpu data from different gpu.
    """

    data_cat = data["0"]
    for i in range(1, len(data)):
        data_cat = torch.cat((data_cat, data[str(i)].cuda(0))) # put data in other gpu to 0, then concat
    return data_cat


def cal_pivot(n_losses, net_type, depth, logger):
    """
    Calculate the inserted layer for additional loss
    """

    num_segments = n_losses + 1 # divide into n_losses + 1 segment
    network_block_num = block_num[net_type] if net_type == 'mobilenet' else block_num[net_type + str(depth)]
    num_block_per_segment = (network_block_num // num_segments) + 1 # evenly insert
    pivot_set = []
    for i in range(num_segments - 1):
        pivot_set.append(min(num_block_per_segment * (i + 1), network_block_num - 1)) # when the whole layers is not enough
    logger.info("pivot set: {}".format(pivot_set))
    return num_segments, pivot_set
