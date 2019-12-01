# Standard Library
import json
from typing import Any, Dict, List, Union

# First Party
from smdebug.core.modes import ModeKeys
from smdebug.core.utils import step_in_range

DEFAULT_SAVE_CONFIG_INTERVAL = 500
DEFAULT_SAVE_CONFIG_START_STEP = 0
DEFAULT_SAVE_CONFIG_END_STEP = None
DEFAULT_SAVE_CONFIG_SAVE_STEPS = []
ALLOWED_PARAMS = ["save_interval", "save_steps", "start_step", "end_step"]


class SaveConfig:
    """Maps modes to SaveConfigMode.

    This is the object to serialize, unserialize, and pass.
    SaveConfigMode should be used when instantiating this class,
    but use SaveConfig as the main unit.
    """

    def __init__(
        self,
        mode_save_configs: Dict[ModeKeys, "SaveConfigMode"] = None,
        save_interval: int = None,
        start_step: int = None,
        end_step: int = None,
        save_steps: List[int] = None,
    ):
        """Pass in a dictionary mapping modes to SaveConfigs. No parsing here.

        If `mode_save_configs` is missing keys, these will be set as `None` and instantiated
        by the SaveManager.

        Parameters:
        mode_save_configs (dict, e.g. {
            ModeKeys.TRAIN: SaveConfigMode,
            ModeKeys.EVAL: SaveConfigMode,
            ModeKeys.PREDICT: SaveConfigMode,
            ModeKeys.GLOBAL: SaveConfigMode
        }).
        """
        # Simple mode, pass in mode-less parameters directly
        if mode_save_configs is None:
            self.mode_save_configs = {
                mode: SaveConfigMode(
                    save_interval=save_interval,
                    start_step=start_step,
                    end_step=end_step,
                    save_steps=save_steps,
                )
                for mode in ModeKeys
            }
        # Advanced mode, specify each mode
        else:
            if not all(
                [
                    isinstance(mode, ModeKeys)
                    and (value is None or isinstance(value, SaveConfigMode))
                    for mode, value in mode_save_configs.items()
                ]
            ):
                raise ValueError(
                    f"Each key,value in mode_save_configs={mode_save_configs} must be of type ModeKey,SaveConfigMode"
                )
            # Populate default SaveConfigMode for each missing ModeKey
            for mode in ModeKeys:
                if mode not in mode_save_configs:
                    mode_save_configs[mode] = None
            # Save the object
            self.mode_save_configs = mode_save_configs

    def get_save_config(self, mode) -> "SaveConfigMode":
        if self.mode_save_configs[mode] is None:
            raise ValueError(f"SaveConfig={self} is not ready. Call SaveManager.prepare() first.")
        return self.mode_save_configs[mode]

    def set_save_config(self, mode: ModeKeys, save_config_mode: "SaveConfigMode") -> None:
        if not isinstance(save_config_mode, SaveConfigMode):
            raise ValueError(f"save_config_mode={save_config_mode} must be type SaveConfigMode")
        self.mode_save_configs[mode] = save_config_mode

    def should_save_step(self, mode, step_num) -> bool:
        return self.get_save_config(mode).should_save_step(step_num)

    def to_json_dict(self) -> Dict:
        # Convert enums to str
        return {
            mode_key.name: save_config_mode.to_json_dict() if save_config_mode else None
            for mode_key, save_config_mode in self.mode_save_configs.items()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict())

    def merge_default_save_config(self, default_save_config):
        """
        Merges save_config with default_save_config.
        Config for any mode not in save_config will be populated
        by copying the one from default_save_config
        """
        for mode in ModeKeys:
            if self.mode_save_configs[mode] is None:
                if default_save_config.mode_save_configs[mode] is not None:
                    self.set_save_config(
                        mode=mode, save_config_mode=default_save_config.get_save_config(mode)
                    )
                else:
                    self.set_save_config(mode=mode, save_config_mode=SaveConfigMode())

    @classmethod
    def from_dict(
        cls, params: Dict[ModeKeys, Any], default_values: Dict[str, Any] = None
    ) -> "SaveConfig":
        """Parses a dict into a SaveConfig object.

        Appropriate formats:
        Dict[str, SaveConfigMode]
        Dict[str, Dict[str, Any]]
        Dict[ModeKeys, SaveConfigMode]
        Dict[ModeKeys, Dict[str, Any]]
        """
        if params is None:
            return None
        if default_values is None:
            default_values = {}
        # Maybe convert strings to enums
        if all(isinstance(key, str) for key, value in params.items()):
            params = {ModeKeys[key]: value for key, value in params.items()}
        # Maybe convert dicts to SaveConfigMode
        if all(value is None or isinstance(value, dict) for key, value in params.items()):
            params = {
                key: SaveConfigMode.from_dict(value, default_values)
                for key, value in params.items()
            }
        return cls(mode_save_configs=params)

    @classmethod
    def from_json(cls, json_str: str) -> "SaveConfig":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_save_config_mode(cls, save_config_mode) -> "SaveConfig":
        """Create a class where all modes correspond to `save_config_mode`."""
        return cls(mode_save_configs={mode: save_config_mode for mode in ModeKeys})

    @classmethod
    def parse(cls, obj) -> "SaveConfig":
        """Does typechecking and creates a SaveConfig object.

        Appropriate formats:
        None
        SaveConfig
        SaveConfigMode
        Dict[ModeKeys, SaveConfigMode]
        Dict[ModeKeys, Dict[str, Any]]
        """
        if obj is None:
            return cls()
        elif isinstance(obj, SaveConfig):
            return obj
        elif isinstance(obj, SaveConfigMode):
            return cls.from_save_config_mode(obj)
        elif isinstance(obj, dict):
            return cls.from_dict(obj)
        else:
            raise TypeError(f"obj={obj} cannot be parsed into a SaveConfig object")

    def __eq__(self, other):
        if not isinstance(other, SaveConfig):
            return NotImplemented
        return all(
            [self.mode_save_configs[mode] == other.mode_save_configs[mode] for mode in ModeKeys]
        )

    def __repr__(self):
        return f"<class SaveConfig: {self.mode_save_configs}>"


class SaveConfigMode:
    """
    Wrapping all the save configuration parameters into this object.
    This would make it easier to set different save configuration for
    different collections and for the base tensors saved.

    This class should not be serialized by itself, only inside of SaveConfig.

    Parameters:
      save_interval (int): Save every n steps.
      save_steps (list of int): Save at all the steps given in this list. Overrides save_interval.
      start_step (int): Save after n steps.
      end_step (int): Stop saving after n steps.
    """

    def __init__(
        self,
        save_interval: Union[int, str] = None,
        start_step: Union[int, str] = None,
        end_step: Union[int, str] = None,
        save_steps: List[int] = None,
    ):
        if save_interval is None:
            self.save_interval = DEFAULT_SAVE_CONFIG_INTERVAL
        else:
            self.save_interval = int(save_interval)

        if save_steps is None:
            self.save_steps = DEFAULT_SAVE_CONFIG_SAVE_STEPS
        else:
            self.save_steps = save_steps

        if start_step is None:
            self.start_step = DEFAULT_SAVE_CONFIG_START_STEP
        else:
            self.start_step = int(start_step)

        if end_step is None:
            self.end_step = DEFAULT_SAVE_CONFIG_END_STEP
        else:
            self.end_step = int(end_step)

        ## DO NOT REMOVE; please make sure that _check & from_json is updated accordingly.
        self._check()

    def _check(self):
        if any([x not in ALLOWED_PARAMS for x in self.__dict__]):
            raise ValueError(f"Params {self.__dict__.keys()} must be in {ALLOWED_PARAMS}")
        if not isinstance(self.save_interval, int):
            raise ValueError(f"save_interval={self.save_interval} must be type(int)")
        if not (
            isinstance(self.save_steps, list) and all([isinstance(x, int) for x in self.save_steps])
        ):
            raise ValueError(f"save_steps={self.save_steps} must be type(list(int))")
        if not isinstance(self.start_step, int):
            raise ValueError(f"start_step={self.start_step} must be type(int)")
        if not (self.end_step is None or isinstance(self.end_step, int)):
            raise ValueError(f"end_step={self.end_step} must be None or type(int)")

    def to_json_dict(self):
        """Be explicit about what keys we return."""
        return {
            "save_interval": self.save_interval,
            "save_steps": self.save_steps,
            "start_step": self.start_step,
            "end_step": self.end_step,
        }

    @classmethod
    def from_dict(cls, params: Dict[str, Any], default_values: Dict[str, Any] = None):
        if params is None:
            return None
        elif not isinstance(params, dict):
            raise TypeError(f"params={params} is not a dict.")
        if default_values is None:
            default_values = {}
        elif not isinstance(default_values, dict):
            raise TypeError(f"default_values={default_values} is not a dict.")
        return cls(
            save_interval=params.get("save_interval", default_values.get("save_interval")),
            start_step=params.get("start_step", default_values.get("start_step")),
            end_step=params.get("end_step", default_values.get("end_step")),
            save_steps=params.get("save_steps", default_values.get("save_steps")),
        )

    def __eq__(self, other):
        if not isinstance(other, SaveConfigMode):
            return NotImplemented
        return (
            self.save_interval == other.save_interval
            and self.save_steps == other.save_steps
            and self.start_step == other.start_step
            and self.end_step == other.end_step
        )

    def should_save_step(self, step_num: int):
        rval = False
        if self.save_steps and step_num in self.save_steps:
            rval = True
        elif (
            step_in_range((self.start_step, self.end_step), step_num)
            and step_num % self.save_interval == 0
        ):
            rval = True
        return rval

    def __repr__(self):
        return (
            f"<class SaveConfig: save_interval={self.save_interval}, save_steps={self.save_steps}, "
            f"start_step={self.start_step}, end_step={self.end_step}>"
        )
