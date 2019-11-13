# Standard Library
import os


class TrialCatalog(object):
    def __init__(self):
        self.trials = {}

    def add_trial(self, trial_name, trial_object):
        self.trials[trial_name] = trial_object

    def get_trial(self, trial_name):
        return self.trials[trial_name]

    def get_trials(self):
        return self.trials.keys()


class LocalTrialCatalog(TrialCatalog):
    def __init__(self, localdir):
        super().__init__()
        self.localdir = localdir

    def list_candidates(self):
        files_and_folders = os.listdir(self.localdir)
        folders = [x for x in files_and_folders if os.path.isdir(os.path.join(self.localdir, x))]
        return folders


"""
class SageMakerTrialCatalog(TrialCatalog):
    def __init__(self,endpoint,port):
        super().__init__()
        self.endpoint = endpoint
        self.port = port
        self.client = InfluxDBClient(host=self.endpoint, port=self.port)
        self.client.switch_database('tornasole_deb')


    def list_candidates(self):
        points = self.client.query(f"select distinct(expid) from execdata")
        res = []
        for p in poinsmd.get_points():
            res.append(p['distinct'])
        return res
"""
