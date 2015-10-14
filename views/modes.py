from utils import get_run_type_visualizer, get_trial_type_visualizer
<<<<<<< HEAD
=======

>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
import logging
import os

class View(object):
    """
    Base class for all types of views.
    """

    def initialize(self, controller):
        raise NotImplementedError('View is an abstract class, this should '
                                  'not be called.')

    # TODO - description of what it does
    def update(self, trial):
        raise NotImplementedError('View is an abstract class, this should '
                                  'not be called.')


class TerminalView(View):
    """
    View which just prints to the terminal.
    """

    def initialize(self, controller):
<<<<<<< HEAD
        self.controller = controller       
=======
        self.controller = controller
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
        
        if not self.controller.restart and not (self.controller.fitness and  self.controller.configuration):
            logging.error('Benchmark and/or configuration script not '
                          'provided, terminating...')
            return
        controller.load_profile_dict() ## TODO - implement this later.. for the moment 
        if self.controller.restart:
            logging.info("Restarting Most Recent Run")
            self.controller.start_run(self.controller.get_most_recent_run_name(), self.controller.fitness, self.controller.configuration, terminal_mode=True)
        else:
            logging.info("Starting Run")
            self.controller.start_run('Default_run', self.controller.fitness, self.controller.configuration, terminal_mode=True)
        #self.controller.visualizer.terminate()
        
    ## Print out run statistics, define a new stats printer
    def update(self, trial=None, run=None, visualize=None):
        if visualize:
<<<<<<< HEAD
            if not (trial is None): 
                self.plot_view = get_trial_type_visualizer(trial.get_trial_type())["default"]
=======
            if not (trial is None):
                self.plot_view = get_trial_type_visualizer(trial.get_trial_type())["default"] 
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
                if trial.get_main_counter() % trial.get_configuration().vis_every_X_steps == 0: ## TODO - its not ideal... rethink it... 
                    snapshot = trial.snapshot()
                    graphdict = self.controller.get_trial_visualization_dict(trial.get_trial_type())
                    snapshot.update(graphdict)
                    self.controller.visualize(snapshot, self.plot_view.render)
<<<<<<< HEAD
            if not (run is None): 
                self.run_view = get_run_type_visualizer(run.get_trial_type())["default"]
=======
            if not (run is None):
                self.run_view = get_run_type_visualizer(run.get_trial_type())["default"] 
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
                logging.info("Generating Report")
                snapshot = run.snapshot()
                snapshot.update(self.controller.get_run_visualization_dict(run.get_run_type()))
                self.controller.visualize(snapshot, self.run_view.render)

class GUIView(View):
    """
    Proper GUI view.
    """

    def initialize(self, controller):
        import wx
        from views.gui.windows import RunWindow
        self.app = wx.App()
        self.window = RunWindow(controller)
        controller.load_profile_dict()
        self.app.MainLoop()

    def update(self, trial=None, run=None, visualize=False):
        import wx
        from views.gui.windows import UpdateEvent
        wx.PostEvent(self.window.GetEventHandler(), UpdateEvent(trial=trial, run=run, visualize=visualize))
