import numpy as np
import itertools as it


def main():

    # Define probs and rewards for each cond
    # ------------------------------------------------------------------------------- # 
    reward = [[] for _ in range(6)]
    prob = [[] for _ in range(6)]

    reward[0] = [[-1, 1], [-1, 1]]
    prob[0] = [[0.2, 0.8], [0.8, 0.2]]

    reward[1] = [[-1, 1], [-1, 1]]
    prob[1] = [[0.3, 0.7], [0.7, 0.3]]

    reward[2] = [[-1, 1], [-1, 1]]
    prob[2] = [[0.4, 0.6], [0.6, 0.4]]

    reward[3] = [[-1, 1], [-1, 1]]
    prob[3] = [[0.5, 0.5], [0.5, 0.5]]

    # reward[4] = [[-1, 1], [-1, 1]]
    # prob[4] = [[0.1, 0.9], [0.9, 0.1]]
    #
    # reward[5] = [[-1, 1], [-1, 1]]
    # prob[5] = [[1, 0], [0, 1]]
    # cond = np.repeat(range(4), 96)
    # tmax = len(cond)
    # max_list = []
    # rand_list = []
    # for i in range(1000):
    #     max_reward = 0
    #     rand_reward = 0
    #
    #     for t in range(tmax):
    #         r = np.array(reward[cond[t]])
    #         p = np.array(prob[cond[t]])
    #         ev1 = sum(r[0] * p[0])
    #         ev2 = sum(r[1] * p[1])
    #         max_reward += max([ev1, ev2])
    #         rand_reward += np.random.choice([ev1, ev2])
    #
    #     print('max', max_reward)
    #     print('rand', rand_reward)
        # i = 0
        # expected_values = [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1]
        # for r1, p1 in zip(reward, prob):
        #     for r2, p2 in zip(r1, p1):
        #         for e in expected_values:
        #             r, p = np.array(r2), np.array(p2)
        #             ev1 = sum(r * p)
        #             ev2 = e
        #             max_reward += max([ev1, ev2]) * 2
        #             rand_reward += np.random.choice([ev1, ev2]) * 2
        #             i += 1
        # print(i)
        # print('max', max_reward)
        # print('rand', rand_reward)
        # max_list.append(max_reward)
        # rand_list.append(rand_reward)
    # print(np.mean(max_list))
    # print(np.mean(rand_list))

    symbol = []
    for _ in range(2):
        for r1, p1 in zip(reward, prob):
            for r2, p2 in zip(r1, p1):
                symbol.append(np.array([r1, p1]))

    rew_1 = []
    rew_2 = []
    for i in range(1000):
        rand_rew = 0
        max_rew = 0
        for s in symbol:
            rand_rew += abs(sum(s[0] * s[1]) - np.random.uniform(-1, 1))
            max_rew += abs(sum(s[0] * s[1]) - sum(s[0] * s[1]))
        rew_1.append(max_rew)
        rew_2.append(rand_rew)

    print(np.mean(rew_1))
    print(np.mean(rew_2))


if __name__ == "__main__":
    main()
