import numpy as np

def test_irregular_sparsity(model):
    """

        :param model: saved re-trained model
        :return:
        """

    total_zeros = 0
    total_nonzeros = 0

    for name, weight in model.named_parameters():
        if "bias" in name:
            continue
        zeros = np.sum(weight.cpu().detach().numpy() == 0)
        total_zeros += zeros
        non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
        total_nonzeros += non_zeros
        zeros = np.sum(weight.cpu().detach().numpy() == 0)
        non_zero = np.sum(weight.cpu().detach().numpy() != 0)
        print("irregular zeros: {}, irregular sparsity is: {:.4f}".format(zeros, zeros / (zeros + non_zero)))

    print("---------------------------------------------------------------------------")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros+total_nonzeros) / total_nonzeros))
    print("===========================================================================\n\n")


def test_column_sparsity(model):
    """

    :param model: saved re-trained model
    :return:
    """

    total_zeros = 0
    total_nonzeros = 0

    total_column = 0
    total_empty_column = 0

    for name, weight in model.named_parameters():
        if(len(weight.size()) == 4): # only consider conv layers
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros
            weight2d = weight.reshape(weight.shape[0], -1)
            column_num = weight2d.shape[1]

            empty_column = np.sum(np.sum(weight2d.cpu().detach().numpy(), axis=0) == 0)
            print("empty column of {} is: {}. column sparsity is: {:.4f}".format(
                name, empty_column, empty_column / column_num))

            total_column += column_num
            total_empty_column += empty_column
    print("---------------------------------------------------------------------------")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("total number of column: {}, empty-column: {}, column sparsity is: {:.4f}".format(
        total_column, total_empty_column, total_empty_column / total_column))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros)/total_nonzeros))

    unused = calculate_unused_weight(model)
    print("only consider conv layers, including unused weight, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros) / (total_nonzeros - unused)))
    print("===========================================================================\n\n")


def calculate_unused_weight(model):
    """
        helper funtion to calculate the corresponding filter add-on sparsity to next layer empty channel

        :param model: saved re-trained model
        :return:
        """

    weight_dict = {} # indexed weight copy
    m = 1 # which layer
    n = 1 # which layer
    counter = 1 # layer counter
    total_unused_number = 0 # result
    flag1 = False # detect sparsity type
    flag2 = False # detect sparsity type

    for name, weight in model.named_parameters(): # calculate total layer
        if (len(weight.size()) == 4):
            weight_dict[counter] = weight
            counter += 1
    counter = counter - 1

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):
            weight3d = weight.reshape(weight.shape[0], weight.shape[1], -1)
            """ calculate unused filter, previous layer filter <= empty channel by column pruning """
            if m != 1:
                empty_channel_index = []
                for i in range(weight3d.size()[1]):
                    non_zero_filter = np.where(weight3d[:, i, :].cpu().detach().numpy().any(axis=1))[0]
                    if non_zero_filter.size == 0:
                        channel_i = weight3d[0, i, :]
                    else:
                        channel_i = weight3d[non_zero_filter[0], i, :]
                    zeros = np.sum(channel_i.cpu().detach().numpy() == 0)
                    channel_empty_ratio = zeros / weight3d.size()[2]
                    if channel_empty_ratio == 1:
                        empty_channel_index.append(i)
                        flag1 = True
                # print(name, empty_channel_index)

                previous_layer = weight_dict[m - 1]
                filter_unused_num = 0
                for filter_index in empty_channel_index:
                    target_filter = previous_layer[filter_index, :, :, :]
                    filter_unused_num += np.sum(target_filter.cpu().detach().numpy() != 0)  # != 0 to calculate sparsity
                total_unused_number += filter_unused_num

            m += 1

            #=====================================================================================#
            """ calculate unused channel, empty filter by filter pruning => next layer channel """
            if n != counter:
                empty_filter_index = []
                for j in range (weight.size()[0]):
                    if np.sum(weight[j, :, :, :].cpu().detach().numpy()) == 0:
                        empty_filter_index.append(j)
                        flag2 = True
                # print(empty_filter_index)
                next_layer = weight_dict[n + 1]
                channel_unused_num = 0
                for channel_index in empty_filter_index:
                    target_channel = next_layer[:, channel_index, :, :]
                    channel_unused_num += np.sum(target_channel.cpu().detach().numpy() != 0)  # != 0 to calculate sparsity
                total_unused_number += channel_unused_num

            n += 1

    if flag1 and not flag2:
        print("your model has column sparsity")
    elif flag2 and not flag1:
        print("your model has filter sparsity")
    elif flag1 and flag2:
        print("your model has column AND filter sparsity")
    print("total unused weight number (column => prev filter / filter => next column): ", total_unused_number)
    return total_unused_number



def test_chanel_sparsity(model):
    """

    :param model: saved re-trained model
    :return:
    """

    for name, weight in model.named_parameters():
        if(len(weight.size()) == 4):
            weight2d = weight.reshape(weight.shape[0], weight.shape[1] , -1)
            """ check channel sparsity based on column sparsity"""
            # print(weight.size(), weight2d.size())

            almost_empty_channel = 0
            for i in range(weight2d.size()[1]):
                channel_i = weight2d[0, i, :]
                print(channel_i)
                zeros = np.sum(channel_i.cpu().detach().numpy() == 0)
                channel_empty_ratio = zeros / weight2d.size()[2]
                if channel_empty_ratio == 1:
                    almost_empty_channel += 1
                # print(zeros, weight2d.size()[2])
                # print(channel_empty_ratio)
            print("({} {}) almost empty channel: {}, total channel: {}. ratio: {}%".format(name, weight.size(),
                almost_empty_channel, weight2d.size()[1], 100.0 * almost_empty_channel / weight2d.size()[1]))






def test_filter_sparsity(model):
    """

    :param model: saved re-trained model
    :return:
    """

    total_zeros = 0
    total_nonzeros = 0

    total_filters = 0
    total_empty_filters = 0

    for name, weight in model.named_parameters():
        if(len(weight.size()) == 4): # only consider conv layers
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros
            empty_filters = 0
            filter_num = weight.size()[0]

            for i in range(filter_num):
                if np.sum(weight[i,:,:,:].cpu().detach().numpy()) == 0:
                    empty_filters += 1
            print("empty filter of {} is: {}. filter sparsity is: {:.4f}".format(
                name, empty_filters, empty_filters / filter_num))

            total_filters += filter_num
            total_empty_filters += empty_filters

    print("---------------------------------------------------------------------------")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("total number of filters: {}, empty-filters: {}, filter sparsity is: {:.4f}".format(
        total_filters, total_empty_filters, total_empty_filters / total_filters))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros) / total_nonzeros))

    unused = calculate_unused_weight(model)
    print("only consider conv layers, including unused weight, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros) / (total_nonzeros - unused)))
    print("===========================================================================\n\n")