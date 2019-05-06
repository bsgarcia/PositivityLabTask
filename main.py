import numpy as np
import itertools as it

from exp.exp import ExperimentGUI


def generate_bloc(ncontext, ntrial_per_context,
                  context_cond_mapping, reward, prob, interleaved=True):
    """

    :param ncontext:
    :param ntrial_per_context:
    :param context_cond_mapping:
    :param reward:
    :param prob:
    :param interleaved:
    :return:
    """
    # ------------------------------------------------------------------------------- # 

    if interleaved:
        context = np.repeat([range(ncontext)], ntrial_per_context, axis=0)
        # shuffle
        for el in context:
            np.random.shuffle(el)
        # flatten the array
        context = context.flatten()
    else:
        context = np.repeat(range(ncontext), ntrial_per_context)

    # prepare arrays
    reward = np.zeros(ncontext, dtype=object)
    prob = np.zeros(ncontext, dtype=object)
    r = np.zeros(ntrial_per_context, dtype=object)
    p = np.zeros(ntrial_per_context, dtype=object)
    options = [0, 1]
    idx_options = np.repeat([[0, 1]], ntrial_per_context, axis=0)
    cond = [context_cond_mapping[i] for i in context]

    for t in range(ntrial_per_context):
        r[t] = np.array(reward[context[t]])
        p[t] = np.array(prob[context[t]])
        np.random.shuffle(idx_options[t])

    return context, cond, r, p, idx_options, options


def main():

    # Define probs and rewards for each cond
    # ------------------------------------------------------------------------------- # 
    reward = [[] for _ in range(4)]
    prob = [[] for _ in range(4)]

    reward[0] = [[-1, 1], [-1, 1]]
    prob[0] = [[0.2, 0.8], [0.8, 0.2]]

    reward[1] = [[-1, 1], [-1, 1]]
    prob[1] = [[0.3, 0.7], [0.7, 0.3]]

    reward[2] = [[-1, 1], [-1, 1]]
    prob[2] = [[0.4, 0.6], [0.6, 0.4]]

    reward[3] = [[-1, 1], [-1, 1]]
    prob[3] = [[0.5, 0.5], [0.5, 0.5]]
    # ------------------------------------------------------------------------------- # 

    ncond = 4
    ncontext = 8
    context_cond_mapping = np.repeat(
        [range(4)], ncontext/ncond, axis=0).flatten()
    ntrial_per_context = 48

    context, cond, r, p, idx_options, options = generate_bloc(
        ncontext=ncontext,
        ntrial_per_context=ntrial_per_context,
        context_cond_mapping=context_cond_mapping,
        reward=reward,
        prob=prob
    )

    img_list = ['a', 'b',
                'c', 'd',
                'e', 'f',
                'g', 'h',
                'i', 'j',
                'k', 'l',
                'm', 'n',
                'o', 'p']

    np.random.shuffle(img_list)

    context_map = {k: tuple(v) for k, v in enumerate(np.random.choice(
        img_list, size=(len(img_list)//2, 2), replace=False
    ))}

    exp = ExperimentGUI(name="RetrieveAndCompare")
    exp.init_phase(
        context_map=context_map,

    )
    exp.run()


if __name__ == "__main__":
    main()
