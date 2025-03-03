from stable_baselines3.common.callbacks import BaseCallback


class LogSnakeLength(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        for i in range(len(self.locals['infos'])):
            if 'terminal_observation' in self.locals['infos'][i].keys():
                #print(self.locals['infos'][i]['score'][0])
                self.logger.record("snake_score", self.locals['infos'][i]['score'][0])
                #self.logger.record("snake_length", self.locals['infos'][i]['terminal_observation']['score'][0])
        return True
