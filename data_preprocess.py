import sys
import pandas as pd
import numpy as np


def data_preprocess(offline_data_path: str, base_days=20):
    """Define user state from original offline data and generate new offline data containing user state
    Args:
        offline_data_path: Path of original offline data
        base_days: The number of historical days required for defining the initial user state

    Return:
        new_offline_data(pd.DataFrame): New offline data containing user state.
            User state, 3 dimension：Historical total order number、historical average of non-zero day order number、and historical average of non-zero day order fee
        user_states_by_day(np.ndarray): An array of shape (number_of_days, number_of_users, 3), contains all the user states.
            If base_days is equal to 30, the number_of_days is equal to total_days - base_days, 30.
            And user_states_by_day[0] means all the user states in the first day.
        evaluation_start_states(np.ndarray): the states for the first day of validation in this competition
    """
    # *****base_day改为20天，可以多获得10天的数据用来学习环境，20天是否合理待定*****
    df = pd.read_csv(offline_data_path)
    # # 测试截取两例
    # df = df.iloc[0:240, :]
    total_days = df['step'].max() + 1  # 60

    # *****加入'day_deliver_coupon_num','coupon_discount'列，目前df里只去除了step和date*****
    useful_df = df[['index', 'day_deliver_coupon_num', 'coupon_discount', 'day_order_num', 'day_average_order_fee']]
    index = np.arange(0, df.shape[0], total_days)  # 标识每个用户初始index0.60.120.....
    initial_data_index = np.array([np.arange(i, i+base_days) for i in index]).flatten()  # 拼成一维数组
    rollout_data_index = np.array([np.arange(i+base_days, i+total_days) for i in index]).flatten()

    initial_data = useful_df.iloc[initial_data_index, :]
    # 总订单数和平均订单数
    day_order_num_df = initial_data.groupby('index')['day_order_num'].agg(total_num='sum', nozero_time=np.count_nonzero)
    day_order_num_df['average_num'] = day_order_num_df.apply(lambda x: x['total_num'] / x['nozero_time'] if x['nozero_time'] > 0 else 0, axis=1)

    # *****加入’优惠券利用率‘状态：计算公式为min(历史总订单数/历史发券数,1),除零检查：历史发券数为0时利用率为1*****
    day_deliver_num_df = initial_data.groupby('index').agg('sum')
    day_deliver_num_df['day_coupon_use'] = day_deliver_num_df.apply(lambda x: min(x['day_order_num'] / x['day_deliver_coupon_num'], 1)
                                                                    if x['day_deliver_coupon_num'] != 0 else 1, axis=1)
    # *****加入用户利用折扣状态，计算公式为用户下单时所用过的平均折扣，未下过单折扣取0.6*****
    discount_use_df = initial_data[(initial_data['day_deliver_coupon_num'] != 0) & (initial_data['day_order_num'] != 0)]
    average_discount_use_df = discount_use_df.groupby('index')['coupon_discount'].agg(average_used_discount='mean')
    average_discount_use_df = average_discount_use_df.reindex(index=[i for i in range(day_order_num_df.shape[0])], fill_value=0.6)

    # 平均订单金额
    day_order_average_fee_df = initial_data.groupby('index')['day_average_order_fee'].agg(total_fee='sum', nozero_time=np.count_nonzero)
    day_order_average_fee_df['average_fee'] = day_order_average_fee_df.apply(lambda x: x['total_fee'] / x['nozero_time']
                                                                             if x['nozero_time'] > 0 else 0, axis=1)
    # *****day_coupon_use, average_used_discount拼入initial_states*****
    initial_states = pd.concat([day_order_num_df[['total_num', 'average_num']], day_order_average_fee_df[['average_fee']],
                                day_deliver_num_df['day_coupon_use'], average_discount_use_df['average_used_discount']], axis=1)

    # *****initial_states归一化*****
    MAX_STATES = np.array([120, 6, 100, 1, 1])

    initial_states = initial_states / MAX_STATES

    traj_num = initial_states.shape[0]  # 1000
    initial_states = initial_states.to_numpy()
    traj_list = []
    final_state_list = []
    for t in range(traj_num):
        state = initial_states[t, :]
        traj_list.append(state)
        index = np.arange(t * total_days, (t+1) * total_days)
        user_act = useful_df.to_numpy()[index, 1:][base_days:]
        for i in range(total_days-base_days):
            cur_act = user_act[i]
            next_state = np.empty(state.shape)
            size = (state[0] * MAX_STATES[0] / (state[1] * MAX_STATES[1])) if state[1] > 0 else 0  # 下单天数
            next_state[0] = state[0] + cur_act[2] / MAX_STATES[0]
            next_state[1] = state[1] + (cur_act[2] - state[1] * MAX_STATES[1]) * float(cur_act[2] > 0.0) / ((size + 1) * MAX_STATES[1])
            next_state[2] = state[2] + (cur_act[3] - state[2] * MAX_STATES[2]) * float(cur_act[3] > 0.0) / ((size + 1) * MAX_STATES[2])
            # *****优惠券利用率，历史使用折扣，上下文中cur_act索引更改, 归一化状态*****
            next_state[3] = min((next_state[0] * MAX_STATES[0]) / ((state[0] * MAX_STATES[0]) / max(state[3], 0.01) + cur_act[0]), 1) \
                if ((state[0] * MAX_STATES[0]) / max(state[3], 0.01) + cur_act[0]) != 0 else 1
            next_state[4] = state[4] + (cur_act[1] - state[4]) * float(cur_act[0] > 0.0) * float(cur_act[2] > 0.0) / (size + 1)

            state = next_state
            traj_list.append(state) if (i + 1) < total_days-base_days else final_state_list.append(state)

    history_state_array = np.array(traj_list)
    evaluation_start_states = np.array(final_state_list)
    coupon_action_array = df.iloc[rollout_data_index, 1:3].to_numpy()
    user_action_array = df.iloc[rollout_data_index, 3:5].to_numpy()
    step_list = [[j] for _ in range(traj_num) for j in range(total_days - base_days)]  # 1000行30列[0][1][29],1000行这样的
    traj_index_list = [[i] for i in range(traj_num) for _ in range(total_days - base_days)]  # 1000行30列[0]30...[1000]30
    steps, trajs = np.array(step_list), np.array(traj_index_list)
    # *****history_state_array中加入'day_coupon_use'*****
    columns = ['index', 'total_num', 'average_num', 'average_fee', 'day_coupon_use', 'average_used_discount', 'day_deliver_coupon_num', 'coupon_discount', 'day_order_num', 'day_average_order_fee', 'step']
    new_offline_data = pd.DataFrame(data=np.concatenate([trajs, history_state_array, coupon_action_array, user_action_array, steps], -1), columns=columns)

    index = np.arange(0, history_state_array.shape[0], total_days-base_days)
    user_states_by_day = np.array([history_state_array[index+i] for i in range(total_days-base_days)])
    new_offline_data_dict = {"state": history_state_array, "action_1": coupon_action_array, "action_2": user_action_array, "index": index + (total_days-base_days) }

    return new_offline_data, user_states_by_day, evaluation_start_states, new_offline_data_dict


if __name__ == "__main__":
    offline_data = sys.argv[1]
    new_offline_data, user_states_by_day, evaluation_start_states, new_offline_data_dict = data_preprocess(offline_data)
    print(new_offline_data.shape)
    print(user_states_by_day.shape)
    print(evaluation_start_states.shape)
    # *****改new_offline_data.to_csv的名称*****
    new_offline_data.to_csv('offline_592_5_dim_state.csv', index=False)
    np.save('user_states_by_day.npy', user_states_by_day)
    np.save('evaluation_start_states.npy', evaluation_start_states)
    np.savez('venv.npz', **new_offline_data_dict)

    # data_preprocess('D:\\staticData\\Data\\project\\competition\\starting_kit\\baseline\\offline_test.csv')
