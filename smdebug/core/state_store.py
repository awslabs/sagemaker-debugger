# Standard Library
import json
import os
import time

# First Party
from smdebug.core.config_constants import (
    CHECKPOINT_CONFIG_FILE_PATH_ENV_VAR,
    CHECKPOINT_DIR_KEY,
    DEFAULT_CHECKPOINT_CONFIG_FILE,
    LATEST_GLOBAL_STEP_SAVED,
    LATEST_GLOBAL_STEP_SEEN,
    LATEST_MODE_STEP,
    METADATA_FILENAME,
    TRAINING_RUN,
)
from smdebug.core.logger import get_logger

logger = get_logger()
# This is 'predicate' for sorting the list of states based on seen steps.
def _rule_for_sorting(state):
    return state[LATEST_GLOBAL_STEP_SEEN]


class StateStore:
    def __init__(self):
        self._saved_states = []
        self._checkpoint_update_timestamp = 0
        self._states_file = None
        self._checkpoint_dir = None
        self._retrieve_path_to_checkpoint()
        if self._checkpoint_dir is not None:
            self._states_file = os.path.join(self._checkpoint_dir, METADATA_FILENAME)
            self._read_states_file()
            self._checkpoint_update_timestamp = max(
                os.path.getmtime(child) for child, _, _ in os.walk(self._checkpoint_dir)
            )

    def _retrieve_path_to_checkpoint(self):
        """
        Retrieve the folder/path where users will store the checkpoints. This path will be stored as a value for key
        'CHECKPOINT_DIR_KEY' in the checkpoint config file.
        We will monitor this folder and write the current state if this folder is recently modified.
        """
        if self._checkpoint_dir is not None:
            return self._checkpoint_dir
        checkpoint_config_file = os.getenv(
            CHECKPOINT_CONFIG_FILE_PATH_ENV_VAR, DEFAULT_CHECKPOINT_CONFIG_FILE
        )
        if os.path.exists(checkpoint_config_file):
            with open(checkpoint_config_file) as json_data:
                parameters = json.load(json_data)
                if CHECKPOINT_DIR_KEY in parameters:
                    self._checkpoint_dir = parameters[CHECKPOINT_DIR_KEY]
        else:
            logger.debug(f"The checkpoint config file {checkpoint_config_file} does not exist.")

    def _read_states_file(self):
        """
        Read the states from the file and create a sorted list of states.
        The states are sorted based on the last seen step.
        """
        if os.path.exists(self._states_file):
            with open(self._states_file) as json_data:
                parameters = json.load(json_data)
            for param in parameters:
                ts_state = dict()
                ts_state[TRAINING_RUN] = param[TRAINING_RUN]
                ts_state[LATEST_GLOBAL_STEP_SAVED] = param[LATEST_GLOBAL_STEP_SAVED]
                ts_state[LATEST_GLOBAL_STEP_SEEN] = param[LATEST_GLOBAL_STEP_SEEN]
                ts_state[LATEST_MODE_STEP] = param[LATEST_MODE_STEP]
                self._saved_states.append(ts_state)
        self._saved_states.sort(key=_rule_for_sorting)

    def is_checkpoint_updated(self):
        """
        Check whether the folder in which checkpoints are stored got updated.
        Update to that folder indicates, user attempted to store the new checkpoint in that directory.
        """
        if self._checkpoint_dir is not None:
            checkpoint_timestamp = max(
                os.path.getmtime(child) for child, _, _ in os.walk(self._checkpoint_dir)
            )
            if checkpoint_timestamp > self._checkpoint_update_timestamp:
                return True
        return False

    def get_last_saved_state(self):
        """
        Retrieve the last save state from the state file if exists.
        The file can contain multiple states. The function will return only the last saves state.
        """
        if len(self._saved_states) > 0:
            return self._saved_states[-1]
        return None

    def update_state(self, ts_state):
        """
        Write the passed state to state file. Since the state file is stored in the same folder as
        that of checkpoints, we update the checkpoint update timestamp after state is written to the file.
        """
        self._saved_states.append(ts_state)
        with open(self._states_file, "w") as out_file:
            json.dump(self._saved_states, out_file)
        self._checkpoint_update_timestamp = time.time()
