import os
import numpy as np
import codecs
import csv
import psychopy as psy
from psychopy import data, core, event, gui, visual


class AbstractExperiment:
    """

    Abstract class that implements logic

    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        #  experiment parameters
        self.ntrial = None
        self.context = None
        self.reward = None
        self.prob = None
        self.idx_options = None
        self.options = None
        self.context_map = None
        self.elicitation_option = None
        self.elicitation_stim = None
        self.cond = None
        self.session = None

        #  init
        self.trial_handler = None
        self.exp_info = None
        self.info_dlg = None
        self.datafile = None
        self.phase = {}

    def init_phase(self, ntrial, cond, context, reward, prob, idx_options,
                   options, context_map, session, elicitation_stim, elicitation_option):
        self.ntrial = ntrial
        self.context = context
        self.reward = reward
        self.prob = prob
        self.idx_options = idx_options
        self.options = options
        self.context_map = context_map
        self.session = session
        self.elicitation_option = elicitation_option
        self.elicitation_stim = elicitation_stim
        self.cond = cond

    def generate_trials(self):
        trial_list = []
        for t in range(self.ntrial):
            trial_list.append({**{'t': t}, **self.exp_info})

        self.trial_handler = psy.data.TrialHandler(
            trial_list, 1, method="sequential"
        )

        return self.trial_handler

    def play(self, t, choice):
        return np.random.choice(
            self.reward[t][int(choice)], p=self.prob[t][int(choice)]
        )

    def write_csv(self, trial_info):

        if not os.path.isfile(self.datafile):
            with codecs.open(self.datafile, 'ab+', encoding='utf8') as f:
                csv.writer(f, delimiter=',').writerow(list(trial_info.keys()))
                csv.writer(f, delimiter=',').writerow(list(trial_info.values()))
        else:
            with codecs.open(self.datafile, 'ab+', encoding='utf8') as f:
                csv.writer(f, delimiter=',').writerow(list(trial_info.values()))


class AbstractGUI(AbstractExperiment):
    """

    Abstract GUI Component

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.win = None
        self.exp_info = None
        self.info_dlg = None
        self.name = None
        self.datafile = None
        self.stim = None

        self.name = kwargs.get('name')
        self.img_list = kwargs.get('img_list')

    def init_experiment_window(self):
        self.win = psy.visual.Window(
            size=(1300, 800),
            fullscr=False,
            screen=0,
            allowGUI=False,
            allowStencil=False,
            monitor='testMonitor',
            color=(-1, -1, -1),
            colorSpace='rgb',
            blendMode='avg',
            winType='pyglet',
            autoDraw=False
        )

    def init_experiment_info(self):

        subject_id = len(os.listdir('data'))

        self.exp_info = {
            'subject_id': subject_id,
            'elicitation': [0, 1, 2],
            'age': '',
            'gender': ['male', 'female'],
            'date': psy.data.getDateStr(format="%Y-%m-%d_%H:%M")
        }

        self.info_dlg = psy.gui.DlgFromDict(
            dictionary=self.exp_info,
            title=self.name,
            fixed=['ExpVersion'],
        )
        self.exp_info['elicitation'] = int(self.exp_info['elicitation'])

        self.datafile = f'data{os.path.sep}\
            subject_{subject_id}_elicit_{self.exp_info["elicitation"]}.csv'

        # try:
        #     f = open(f'data/{self.datafile}')
        #     arr = np.genfromtxt(f'data/{self.datafile}', delimiter=',', skip_header=True)
        #     self.exp_info['subject_id'] = max(arr[:, 0])
        # except FileNotFoundError:
        #     self.exp_info['subject_id'] = 0

        if self.info_dlg.OK:
            return self.exp_info

        psy.core.quit()

    @staticmethod
    def create_text_stimulus(win, text, height, color):

        text = psy.visual.TextStim(
            win=win,
            ori=0,
            text=text,
            font='Arial',
            height=height,
            color=color,
            colorSpace='rgb',
            alignHoriz='center',
            alignVert='center'
        )
        return text

    @staticmethod
    def create_text_box_stimulus(win, pos, boxcolor='white', outline='grey'):
        rect = psy.visual.Rect(
            win=win,
            width=.25,
            height=.25,
            fillColor=boxcolor,
            lineColor=outline,
            pos=pos,
        )
        return rect

    @staticmethod
    def create_rating_scale(win, pos):
        # rating scale
        scale = visual.RatingScale(
            win, low=-1, high=1, size=1.5, precision=10,
            tickMarks=[('%.1f' % i) for i in np.linspace(-1, 1, 21)],
            tickHeight=0.6,
            markerStart='0', marker='slider', markerColor='white',
            textSize=.25, showValue=True, textColor='white',
            acceptSize=1.5,
            showAccept=False, noMouse=True, maxTime=1000, pos=pos)
        scale.marker.setSize(0.3)
        scale.line.setLineWidth(0.7)
        # scale.accept.setColor('black')
        return scale

    @staticmethod
    def present_stimulus(obj, pos=None, size=None):
        if size:
            obj.setSize(size)
        if pos:
            obj.setPos(pos)
        obj.draw()

    @staticmethod
    def get_keypress():
        try:
            return psy.event.getKeys()[0]
        except IndexError:
            pass

    @staticmethod
    def wait_for_lr_response():
        res = None
        key_list = ('left', 'right')
        while res not in key_list:
            res = psy.event.waitKeys(keyList=key_list)[0]
        return res

    @staticmethod
    def make_dir(dirname):
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    @staticmethod
    def get_files(path='./resources'):
        return [file for i, j, file in os.walk(path)][0]

    @staticmethod
    def load_files(win, files, path='resources/'):
        stim = {}
        for filename in sorted(files):

            ext = filename[-3:]
            name = filename[:-4]

            if ext in ('bmp', 'jpg', 'png'):
                stim[name] = psy.visual.ImageStim(
                    win, image=f'{path}{filename}', color='white'
                )

            elif ext == 'txt':

                with codecs.open(f'{path}{filename}', 'r') as f:
                    stim[name] = psy.visual.TextStim(
                        win,
                        text=f.read(),
                        wrapWidth=1.2,
                        alignHoriz='center',
                        alignVert='center',
                        height=0.20
                    )
        return stim

    def escape(self):
        if self.get_keypress() == 'escape':
            self.win.close()
            psy.core.quit()

    def init(self):

        self.make_dir('data')

        # Set experiment infos (subject age, id, date etc.)
        self.init_experiment_info()

        if self.exp_info is None:
            print('User cancelled')
            psy.core.quit()

        # Show exp window
        self.init_experiment_window()

        # Load files
        names = self.get_files()
        self.stim = self.load_files(win=self.win, files=names)


class ExperimentGUI(AbstractGUI):
    """

    GUI component

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #  experiment parameters
        self.pos_left = [-0.3, 0]
        self.pos_right = [0.3, 0]
        self.pos_up = [0, 0.3]
        self.pos_down = [0, -0.3]

    def display_selection(self, left_or_right):
        pos = [self.pos_left, self.pos_right][left_or_right][:]
        pos[1] -= 0.25
        self.present_stimulus(self.stim['arrow'], pos=pos, size=(0.04, 0.07))

    def display_rating_scale(self, t):
        # rating scale
        scale = self.create_rating_scale(win=self.win, pos=self.pos_down)

        while scale.noResponse:
            scale.draw()

            self.present_stimulus(
                self.create_text_stimulus(
                    win=self.win,
                    text='Selectionnez la valeur avec les flèches gauche et droite\n'
                         'puis validez en appuyant sur Entrée',
                    height=0.04,
                    color='white'
                ),
                pos=(0, -0.65))
            self.present_stimulus(
                self.create_text_stimulus(
                    win=self.win,
                    text=('%.1f' % scale.getRating()),
                    height=0.09,
                    color='white'
                ),
                pos=(0, -0.5)
            )
            self.display_single(t)
            self.win.flip()

        return float('%.1f' % scale.getRating())

    def display_fixation(self):
        self.present_stimulus(self.stim['cross'])

    def display_welcome(self):
        self.stim['welcome'].setHeight(0.06)
        self.present_stimulus(self.stim['welcome'])

    def display_end(self):
        self.present_stimulus(self.stim['end'])

    def display_pause(self):
        self.present_stimulus(self.stim['pause'])

    def display_counterfactual_outcome(self, outcomes, choice, t, color='red'):
        pos, text = [self.pos_left, self.pos_right], [None, None]

        # set order
        cf_out = outcomes[not choice]
        out = outcomes[choice]
        text[choice] = \
            f'+{out} €' if out > 0 else f'{out} €'
        text[not choice] = \
            f'+{cf_out} €' if cf_out > 0 else f'{cf_out} €'
        text = np.array(text, dtype=str)[self.idx_options[t]]

        # Display
        for t, p in zip(text, pos):
            self.present_stimulus(
                self.create_text_stimulus(win=self.win, text=t, color=color, height=0.13), pos=p
            )

    def display_outcome(self, outcome, left_or_right, color='red'):
        pos = [self.pos_left, self.pos_right][left_or_right][:]
        text = f'+{outcome} €' if outcome > 0 else f'{outcome} €'
        self.present_stimulus(
            self.create_text_stimulus(win=self.win, text=text, color=color, height=0.13),
            pos=pos
        )

    def display_pair(self, t):
        left, right = np.array(
            self.context_map[self.context[t]])[self.idx_options[t]]
        self.present_stimulus(self.stim[left], pos=self.pos_left, size=0.25)
        self.present_stimulus(self.stim[right], pos=self.pos_right, size=0.25)

    def display_exp_desc_pair(self, t, elicitation):

        img, text = self.context_map[self.context[t]]
        p = self.prob[t][self.elicitation_option[t]]
        r = self.reward[t][self.elicitation_option[t]]

        pwin = p[1]
        plose = p[0]
        rwin = r[1]
        rlose = r[0]

        if elicitation:
            txt = ('%.1f pts' % sum([pwin*rwin, plose*rlose]))
        else:
            txt = f'{int(pwin*100)}% chance of winning {rwin}\n\n' \
                f'{int(plose*100)}% chance of losing {rlose}'

        text = self.create_text_stimulus(
            self.win, text=txt, color='white', height=.037)
        textbox = self.create_text_box_stimulus(
            self.win, boxcolor='black', outline='white', pos=self.pos_left)

        self.present_stimulus(self.stim[img], pos=self.pos_right, size=0.25)
        self.present_stimulus(textbox, size=1.1)
        self.present_stimulus(text, pos=self.pos_left)

    def display_single(self, t, pos=None):
        img = self.img_list[self.elicitation_stim[t]]
        self.present_stimulus(self.stim[img], pos=pos if pos else self.pos_up, size=0.25)

    def display_time(self, t):
        self.present_stimulus(self.create_text_stimulus(
            self.win, text=str(t), color='white', height=0.12), pos=(0.7, 0.8)
        )

    def display_continue(self):
        self.present_stimulus(
            self.create_text_stimulus(
                self.win,
                text='Pressez la barre espace pour continuer.',
                color='white',
                height=0.07
            ),
            pos=(0, -0.4)
        )

    def run_trials(self, trial_obj):

        timer = psy.core.Clock()
        self.win.flip()

        for trial in trial_obj:
            # Check if escape key is pressed
            self.escape()
            t = trial['t']
            # Check if a pause is programmed
            #self.check_for_pause(t)

            # Fixation
            self.display_time(t)
            self.display_fixation()
            self.win.flip()
            psy.core.wait(0.2)
            self.display_time(t)
            self.display_fixation()
            self.display_pair(t)
            self.win.flip()

            # Reset timer
            timer.reset()

            res = self.wait_for_lr_response()
            reaction_time = timer.getTime()
            pressed_right = res == 'right'

            c = self.options[
                self.idx_options[t][int(pressed_right)]
            ]

            # Test if choice has a superior expected utility
            superior = sum(self.reward[t][c] * self.prob[t][c]) > \
                       sum(self.reward[t][int(not c)] * self.prob[t][int(not c)])
            # Test for equal utilities
            equal = sum(self.reward[t][c] * self.prob[t][c]) == \
                    sum(self.reward[t][int(not c)] * self.prob[t][int(not c)])

            # Fill trial object
            trial['reaction_time'] = reaction_time
            trial['choice'] = c + 1
            trial['choice_maximizing_utility'] = 1 if superior else 0 if not equal else -1
            trial['probabilities'] = self.prob[t]
            trial['rewards'] = self.reward[t]
            trial['outcome'] = self.play(t=t, choice=c)
            trial['cf_outcome'] = self.play(t=t, choice=not c)
            trial['key_pressed'] = res
            trial['elicitation'] = -1
            trial['cond'] = self.cond[t]
            trial['session'] = self.session

            self.display_time(t)
            self.display_fixation()
            self.display_pair(t)
            self.display_selection(left_or_right=pressed_right)
            self.win.flip()
            psy.core.wait(0.2)

            self.display_outcome(
                outcome=trial['outcome'],
                left_or_right=pressed_right
            )

            self.display_time(t)
            self.display_fixation()
            self.display_selection(left_or_right=pressed_right)
            self.win.flip()
            psy.core.wait(0.5)

            self.write_csv(trial)

    def run_post_test(self, trial_obj):

        timer = psy.core.Clock()
        self.win.flip()

        for trial in trial_obj:

            res = None
            t = trial['t']

            # Fixation
            self.display_time(t)
            self.display_fixation()
            self.win.flip()
            psy.core.wait(0.5)
            self.display_time(t)
            self.display_fixation()

            if self.exp_info['elicitation'] in (0, 1):
                self.display_exp_desc_pair(
                    t,
                    elicitation=self.exp_info['elicitation']
                )
            else:
                res = self.display_rating_scale(t)

            self.win.flip()

            # Reset timer
            timer.reset()

            if not res:
                res = self.wait_for_lr_response()

            reaction_time = timer.getTime()

            if self.exp_info['elicitation'] in (0, 1):
                pressed_right = res == 'right'

                c = self.options[
                    self.idx_options[t][int(pressed_right)]
                ]
                # Test if choice has a superior expected utility
                superior = sum(self.reward[t][c] * self.prob[t][c]) > \
                           sum(self.reward[t][int(not c)] * self.prob[t][int(not c)])
                # Test for equal utilities
                equal = sum(self.reward[t][c] * self.prob[t][c]) == \
                        sum(self.reward[t][int(not c)] * self.prob[t][int(not c)])
                trial['reaction_time'] = reaction_time
                trial['choice'] = c + 1
                trial['choice_maximizing_utility'] = 1 if superior else 0 if not equal else -1
                trial['probabilities'] = self.prob[t]
                trial['rewards'] = self.reward[t]
                trial['outcome'] = self.play(t=t, choice=c)
                trial['cf_outcome'] = self.play(t=t, choice=not c)
                trial['key_pressed'] = res
                trial['elicitation'] = self.exp_info['elicitation']

            else:
                # Fill trial object
                trial['reaction_time'] = reaction_time
                trial['choice'] = res
                trial['choice_maximizing_utility'] = -1
                trial['probabilities'] = -1
                trial['rewards'] = -1
                trial['outcome'] = -1
                trial['cf_outcome'] = -1
                trial['key_pressed'] = -1
                trial['elicitation'] = self.exp_info['elicitation']
            trial['cond'] = self.cond[t]
            trial['session'] = self.session
            self.display_time(t)
            self.display_fixation()

            if self.exp_info['elicitation'] in (0, 1):
                self.display_exp_desc_pair(t, elicitation=self.exp_info['elicitation'])
                self.display_selection(left_or_right=pressed_right)

            self.win.flip()
            psy.core.wait(0.6)

            if self.exp_info['elicitation'] in (0, 1):
                self.display_outcome(
                    outcome=trial['outcome'],
                    left_or_right=pressed_right
                )

            self.display_time(t)
            self.display_fixation()
            if self.exp_info['elicitation'] in (0, 1):
                self.display_selection(left_or_right=pressed_right)
            self.win.flip()
            psy.core.wait(0.5)
            self.write_csv(trial)

    def run(self, welcome=False, post_test=False):

        if welcome:
            # Display greetings
            self.display_welcome()
            self.win.flip()

            psy.event.waitKeys()
            psy.event.clearEvents()

        if post_test:
            self.run_post_test(self.generate_trials())
        else:
            self.run_trials(self.generate_trials())

    def end(self):
        self.display_end()
        self.win.flip()

        psy.event.waitKeys()
        psy.core.quit()


if __name__ == '__main__':
    exit('Please run the main.py script')
