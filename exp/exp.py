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

    def __init__(self, ntrial, context, reward,
                 prob, context_map, idx_options, options, pause,
                 *args, **kwargs):
        super().__init__()

        #  experiment parameters
        self.ntrial = ntrial
        self.context = context
        self.reward = reward
        self.prob = prob
        self.idx_options = idx_options
        self.options = options
        self.pause = pause
        self.context_map = context_map
        self.post_test_stims = np.array(list(context_map.values()), dtype=object).flatten()
        self.context_post = np.random.randint(8, size=10)

        #  init
        self.trial_handler = None
        self.exp_info = None
        self.info_dlg = None
        self.datafile = None

    def generate_trials(self):
        trial_list = []
        for t in range(self.ntrial):
            trial_list.append({
                't': t
            })
            trial_list[t].update(self.exp_info)

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


class AbstractGUI:
    """

    Abstract GUI Component

    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.win = None
        self.exp_info = None
        self.info_dlg = None
        self.name = None
        self.datafile = None
        self.img = None
        self.txt = None
        if kwargs.get('name'):
            self.name = kwargs.get('name')

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

        self.exp_info = {
            'subject_id': '',
            'session:': '',
            'elicitation': ['0', '1', '2'],
            'age': '',
            'gender': ['male', 'female'],
            'date': psy.data.getDateStr(format="%Y-%m-%d_%H:%M")
        }

        self.datafile = f'data{os.path.sep}\
            {self.exp_info["elicitation"]}.csv'

        try:
            f = open(f'data/{self.datafile}')
            arr = np.genfromtxt(f'data/{self.datafile}', delimiter=',', skip_header=True)
            self.exp_info['subject_id'] = max(arr[:, 0])
        except FileNotFoundError:
            self.exp_info['subject_id'] = 0

        self.info_dlg = psy.gui.DlgFromDict(
            dictionary=self.exp_info,
            title=self.name,
            fixed=['ExpVersion'],
        )

        if self.info_dlg.OK:
            return self.exp_info

    @staticmethod
    def create_text_stimulus(win, text, height, color):
        text = psy.visual.TextStim(
            win=win,
            ori=0,
            text=text,
            font='Arial',
            height=height,
            color=color,
            colorSpace='rgb'
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
            win, low=-1, high=1, size=1, precision=10, tickMarks=['-1', '1'],
            markerStart='0', marker='circle', textSize=.5, showValue=True,
            showAccept=True, noMouse=True, maxTime=1000, pos=pos)
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
    def wait_for_response():
        key = psy.event.waitKeys(keyList=['left', 'right'])[0]
        while key not in ['left', 'right']:
            key = psy.event.waitKeys(keyList=['left', 'right'])[0]
        return key

    @staticmethod
    def make_dir(dirname):
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    @staticmethod
    def get_files(path='./resources'):
        return [file for i, j, file in os.walk(path)][0]

    @staticmethod
    def load_files(win, files, path='resources/'):
        img, txt = {}, {}
        for filename in sorted(files):

            ext = filename[-3:]
            name = filename[:-4]

            if ext in ('bmp', 'jpg', 'png'):
                img[name] = psy.visual.ImageStim(
                    win, image=f'{path}{filename}', color='white'
                )

            elif ext == 'txt':

                with codecs.open(f'{path}{filename}', 'r') as f:
                    txt[name] = psy.visual.TextStim(
                        win,
                        text=f.read(),
                        wrapWidth=1.2,
                        alignHoriz='center',
                        alignVert='center',
                        height=0.06
                    )
        return img, txt

    def escape(self):
        if self.get_keypress() == 'escape':
            self.win.close()
            psy.core.quit()

    def run(self):

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
        self.img, self.txt = self.load_files(win=self.win, files=names)


class ExperimentGUI(AbstractExperiment, AbstractGUI):
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
        self.present_stimulus(self.img['arrow'], pos=pos, size=(0.04, 0.07))

    def display_rating_scale(self, t):
        # rating scale
        scale = self.create_rating_scale(win=self.win, pos=self.pos_down)

        while scale.noResponse:
            scale.draw()
            self.display_single(t)
            self.win.flip()

    def display_fixation(self):
        self.present_stimulus(self.img['cross'])

    def display_welcome(self):
        self.present_stimulus(self.txt['welcome'])

    def display_end(self):
        self.present_stimulus(self.txt['end'])

    def display_pause(self):
        self.present_stimulus(self.txt['pause'])

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
        for i, j in zip(text, pos):
            self.present_stimulus(
                self.create_text_stimulus(win=self.win, text=i, color=color, height=0.13), pos=j
            )

    def display_outcome(self, outcome, left_or_right, color='red'):
        pos = [self.pos_left, self.pos_right][left_or_right][:]
        text = f'+{outcome} €' if outcome > 0 else f'{outcome} €'
        self.present_stimulus(
            self.create_text_stimulus(win=self.win, text=text, color=color, height=0.13),
            pos=pos
        )

    def display_pair(self, t):
        img_left, img_right = np.array(
            self.context_map[self.context[t]])[self.idx_options[t]]
        self.present_stimulus(self.img[img_left], pos=self.pos_left, size=0.25)
        self.present_stimulus(self.img[img_right], pos=self.pos_right, size=0.25)

    def display_exp_desc_pair(self, t):
        img, text = self.context_map[self.context_post[t]]
        text = self.create_text_stimulus(
            self.win, text='50% tamere \n50% tonpere', color='white', height=.05)
        textbox = self.create_text_box_stimulus(
            self.win, boxcolor='black', outline='white', pos=self.pos_left)
        self.present_stimulus(self.img[img], pos=self.pos_right, size=0.25)
        self.present_stimulus(textbox)
        self.present_stimulus(text, pos=self.pos_left)

    def display_single(self, t, pos=None):
        img = self.post_test_stims[t]
        self.present_stimulus(self.img[img], pos=pos if pos else self.pos_up, size=0.25)

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

    def check_for_pause(self, t):
        if t == self.pause:
            self.display_pause()
            self.win.flip()
            psy.event.waitKeys()

    def run_trials(self, trial_obj):

        timer = psy.core.Clock()
        self.win.flip()

        for trial in trial_obj:
            # Check if escape key is pressed
            self.escape()
            t = trial['t']
            # Check if a pause is programmed
            self.check_for_pause(t)

            # Fixation
            self.display_time(t)
            self.display_fixation()
            self.win.flip()
            psy.core.wait(0.5)
            self.display_time(t)
            self.display_fixation()
            self.display_pair(t)
            self.win.flip()

            # Reset timer
            timer.reset()

            res = self.wait_for_response()
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
            trial['reaction_time'] = timer.getTime()
            trial['choice'] = c + 1
            trial['choice_maximizing_utility'] = 1 if superior else 0 if not equal else -1
            trial['probabilities'] = self.prob[t]
            trial['rewards'] = self.reward[t]
            trial['outcome'] = self.play(t=t, choice=c)
            trial['cf_outcome'] = self.play(t=t, choice=not c)
            trial['key_pressed'] = res

            self.display_time(t)
            self.display_fixation()
            self.display_pair(t)
            self.display_selection(left_or_right=pressed_right)
            self.win.flip()
            psy.core.wait(0.6)

            self.display_outcome(
                outcome=trial['outcome'],
                left_or_right=pressed_right
            )

            self.display_time(t)
            self.display_fixation()
            self.display_selection(left_or_right=pressed_right)
            self.win.flip()
            psy.core.wait(3)

            self.write_csv(trial)

    def run_post_test(self, trial_obj):

        self.win.flip()

        for t in range(10):

            # Fixation
            self.display_time(t)
            self.display_fixation()
            self.win.flip()
            psy.core.wait(0.5)
            self.display_time(t)
            self.display_fixation()
            self.display_exp_desc_pair(t)
            self.win.flip()
            res = self.wait_for_response()
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

            self.display_time(t)
            self.display_fixation()
            self.display_exp_desc_pair(t)
            self.display_selection(left_or_right=pressed_right)
            self.win.flip()
            psy.core.wait(0.6)

            self.display_outcome(
                outcome=1,
                left_or_right=pressed_right
            )

            self.display_time(t)
            self.display_fixation()
            self.display_selection(left_or_right=pressed_right)
            self.win.flip()
            psy.core.wait(3)

    def run(self):
        super().run()
        # Display greetings
        self.display_welcome()
        self.win.flip()

        psy.event.waitKeys()
        psy.event.clearEvents()

        self.run_post_test([])
        self.run_trials(self.generate_trials())

        self.display_end()
        self.win.flip()

        psy.event.waitKeys()
        psy.core.quit()


if __name__ == '__main__':
    exit('Please run the main.py script')
