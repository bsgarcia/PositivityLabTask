import numpy as np
import itertools as it

from exp.exp import ExperimentGUI


def generate_bloc(ntrial, ncontext, ntrial_per_context,
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
    # reward = np.zeros(ncontext, dtype=object)
    # prob = np.zeros(ncontext, dtype=object)
    r = np.zeros(ntrial, dtype=object)
    p = np.zeros(ntrial, dtype=object)
    options = [0, 1]
    idx_options = np.repeat([[0, 1]], ntrial, axis=0)
    cond = [context_cond_mapping[i] for i in context]

    for t in range(ntrial):
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

    # reward[3] = [[-1, 1], [-1, 1]]
    # prob[3] = [[0.5, 0.5], [0.5, 0.5]]

    reward[3] = [[-1, 1], [-1, 1]]
    prob[3] = [[0.1, 0.9], [0.9, 0.1]]

    # reward[5] = [[-1, 1], [-1, 1]]
    # prob[5] = [[1, 0], [0, 1]]

    ncond = 4
    ncontext = 4
    context_cond_mapping = np.repeat(
        [range(4)], ncontext/ncond, axis=0).flatten()
    ntrial_per_context = 1

    ntrial = ntrial_per_context*ncontext

    context, cond, r, p, idx_options, options = generate_bloc(
        ntrial=ntrial,
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

    #img_list = [i.upper() for i in img_list]

    np.random.shuffle(img_list)

    context_map = {k: tuple(v) for k, v in enumerate(np.random.choice(
        img_list, size=(len(img_list)//2, 2), replace=False
    ))}

    # Start experiment
    # ------------------------------------------------------------------------------- # 
    exp = ExperimentGUI(name="RetrieveAndCompare", img_list=img_list)
    exp.init()
    # # import string
    # # from psychopy import core
    # # for txt in string.ascii_uppercase:
    # #     text = exp.create_text_stimulus(
    # #         exp.win, text=txt, color='black', height=1.2)
    # #     textbox = exp.create_text_box_stimulus(
    # #         exp.win, boxcolor='white', outline='black', pos=(0, 0), linewidth=6)
    # #
    # #     # self.present_stimulus(self.stim[img], pos=self.pos_right, size=0.25)
    # #     exp.present_stimulus(textbox, size=7.9)
    # #     exp.present_stimulus(text, pos=(0, 0))
    # #
    # #     exp.win.flip()
    # #     exp.win.getMovieFrame()  # Defaults to front buffer, I.e. what's on screen now.
    # #     # exp.win.saveMovieFrames(f'{txt}.jpg')
    # #     core.wait(1)
    #
    # from psychopy import core
    # for txt in [('%.1f pts') % (float(i)) for i in (-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1)]:
    #     text = exp.create_text_stimulus(
    #         exp.win, text=txt, color='black', height=0.53, wrapwidth=3)
    #     textbox = exp.create_text_box_stimulus(
    #         exp.win, boxcolor='white', outline='black', pos=(0, 0), linewidth=6)
    #
    #     # self.present_stimulus(self.stim[img], pos=self.pos_right, size=0.25)
    #     exp.present_stimulus(textbox, size=7.9)
    #     exp.present_stimulus(text, pos=(0, 0))
    #     #
    #     exp.win.flip()
    #     exp.win.getMovieFrame()  # Defaults to front buffer, I.e. what's on screen now.
    #     exp.win.saveMovieFrames(f'{txt.replace(" ", "_")}.jpg')
    #     core.wait(1)
    # #
    # from psychopy import core
    # for r, p in zip(reward, prob):
    #     for i, j in zip(r, p):
    #         pwin = j[1]
    #         rwin = i[1]
    #         plose = j[0]
    #         rlose = i[0]
    #         txt = f'{int(pwin*100)}% chance of winning {rwin}\n\n' \
    #             f'{int(plose*100)}% chance of losing {rlose}'
    #
    #         text = exp.create_text_stimulus(
    #             exp.win, text=txt, color='black', height=0.15, wrapwidth=3)
    #         textbox = exp.create_text_box_stimulus(
    #             exp.win, boxcolor='white', outline='black', pos=(0, 0), linewidth=6)
    #
    #     # self.present_stimulus(self.stim[img], pos=self.pos_right, size=0.25)
    #         exp.present_stimulus(textbox, size=7.9)
    #         exp.present_stimulus(text, pos=(0, 0))
    #
    #         exp.win.flip()
    #         exp.win.getMovieFrame()  # Defaults to front buffer, I.e. what's on screen now.
    #         exp.win.saveMovieFrames(f'{pwin}_{plose}.jpg')
    #         core.wait(1)
    #
    #
    #

    # TRAINING
    # ------------------------------------------------------------------------------- # 

    # # Experiment phase
    # exp.init_phase(
    #     ntrial=ntrial,
    #     context=context,
    #     cond=cond,
    #     context_map=context_map,
    #     reward=r,
    #     prob=p,
    #     idx_options=idx_options,
    #     options=options,
    #     session=0,
    #     elicitation_stim=None,
    #     elicitation_option=None
    # )
    #
    # exp.run(welcome=True)

    # Elicitation phase
    exp.init_phase(
        ntrial=ntrial,
        context=context,
        cond=cond,
        context_map=context_map,
        reward=r,
        prob=p,
        idx_options=idx_options,
        options=options,
        session=1,
        elicitation_stim=np.random.randint(len(img_list), size=ntrial),
        elicitation_option=np.random.randint(2, size=ntrial)
    )
    exp.run(welcome=False, post_test=True)
    # ------------------------------------------------------------------------------- # 


if __name__ == "__main__":
    main()
