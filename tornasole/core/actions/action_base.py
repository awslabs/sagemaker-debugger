from tornasole.core.logger import get_logger

class Action:
    def __init__(self):
        self.logger = get_logger()

    def run(self, rule_name, **kwargs):
        pass
