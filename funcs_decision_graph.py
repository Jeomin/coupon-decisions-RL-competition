import torch
import numpy as np
from typing import Dict


def get_next_state(inputs: Dict[str, torch.Tensor]):
    cur_states = inputs['state'].double()
    action_1 = inputs['action_1'].double()  # action_1 policy action
    action_2 = inputs['action_2'].double()  # action_2 user action
    if len(list(action_1.shape)) > 1:
        coupon_num, discount = action_1[:, 0:1], action_1[:, 1:2]
    else:
        length = list(cur_states.shape)[0]
        coupon_num, discount = get_tensor(action_1[0], length), get_tensor(action_1[1], length)
    if len(list(action_2.shape)) > 1:
        day_order_num, day_avg_fee = action_2[:, 0:1], action_2[:, 1:2]
    else:
        length = list(cur_states.shape)[0]
        day_order_num, day_avg_fee = get_tensor(action_2[0], length), get_tensor(action_2[1], length)

    MAX_STATES = torch.tensor([120., 6., 100., 1., 1.]).double()
    next_state = torch.zeros(cur_states.shape, dtype=torch.double)
    size_array = np.array([(x * MAX_STATES[0].item()) / (y * MAX_STATES[1].item()) if y > 0 else 0
                           for i, (x, y) in enumerate(zip(cur_states[:, 0:1].numpy(), cur_states[:, 1:2].numpy()))])
    size_array = torch.tensor(size_array.astype(float)).double()
    print(size_array.shape)

    next_state[:, 0:1] = cur_states[:, 0:1] + day_order_num / MAX_STATES[0].item()
    # next_state[:, 1:2] = cur_states[:, 1:2] + 1 / ((size_array + 1) * MAX_STATES[1].item()) * (day_order_num - cur_states[:, 1:2] * MAX_STATES[1].item()) * (day_order_num > 0.0).double()
    # next_state[:, 2:3] = cur_states[:, 2:3] + 1 / ((size_array + 1) * MAX_STATES[2].item()) * (day_avg_fee - cur_states[:, 2:3] * MAX_STATES[2].item()) * (day_avg_fee > 0.0).double()
    a = torch.ones(next_state[:, 3:4].shape).double()
    next_state[:, 3:4] = torch.minimum((next_state[:, 0:1] * MAX_STATES[0].item()) / ((cur_states[:, 0:1] * MAX_STATES[0].item()) / torch.maximum(cur_states[:, 3:4], a) + coupon_num), a) if ((cur_states[:, 0:1] * MAX_STATES[0].item()) / torch.maximum(cur_states[:, 3:4], a) + coupon_num).all() != 0 else 1
    # next_state[:, 4:5] = cur_states[:, 4:5] + (discount - cur_states[:, 4:5]) * (coupon_num > 0.0).double() * (day_order_num > 0.0).double() / (size_array + 1)

    inputs['state'] = next_state
    return inputs['state']


def get_tensor(data: torch.Tensor, length):
    data = data.item()
    return torch.full([length, 1], data).double()