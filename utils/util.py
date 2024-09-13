import numpy as np
import torch


def trans_np(l_channel):
    params_origin = []
    image_avg_l = np.mean(l_channel)
    image_std_l = np.std(l_channel)
    params_origin.append(image_avg_l)
    params_origin.append(image_std_l)
    params_origin = np.array(params_origin)

    return params_origin


class color_trans(object):
    def __init__(self):
        super(color_trans, self).__init__()

    def  trans_l(self, origin_l, params_origin, params_predict):
        batch_size, channel, _, _ = origin_l.size() #b*1*h*w
        for m in range(batch_size):
            for n in range(channel):
                sd = 1 + n
                t = origin_l[m, n, :, :]
                # color transfer
                mean1 = torch.sub(t, params_origin[m, n])
                std_st = torch.div(params_predict[m, sd], params_origin[m, sd])
                out1 = torch.mul(mean1, std_st)
                out2 = torch.add(out1, params_predict[m, n])
                origin_l[m, n, :, :] = out2
        # get trans_l_channel:b*1*h*w
        return origin_l
