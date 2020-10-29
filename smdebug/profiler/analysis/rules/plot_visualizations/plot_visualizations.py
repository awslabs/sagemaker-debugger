# First Party
# Standard Library
import json
import os
import time
from builtins import Exception

# Third Party
import papermill as pm
from nbconvert.exporters import HTMLExporter
from nbconvert.writers import FilesWriter
from traitlets.config import Config

from smdebug.analysis.utils import parse_bool
from smdebug.core.modes import ModeKeys
from smdebug.core.utils import ensure_dir
from smdebug.exceptions import NoMoreData
from smdebug.rules.rule import Rule
from smdebug.trials import create_trial


class ProcessingJobConfig:
    def __init__(self, processing_job_config_path="/opt/ml/config/processingjobconfig.json"):
        try:
            with open(processing_job_config_path, "r") as tc:
                self.job_config = json.load(tc)
                print(f"job_config:{self.job_config}")
        except Exception as e:
            print(
                f"Got exception while initing ProcessingJobConfig with path {processing_job_config_path}"
            )
        self.job_config_processing_inputs_key = "ProcessingInputs"
        self.job_config_input_name_key = "InputName"
        self.job_config_s3_input_key = "S3Input"
        self.job_config_s3_path = "S3Uri"
        self.base_trial_input_channel_name = "Tensors"

    def get_s3_input_path(self):
        s3_path = ""
        try:
            analytics_inputs = self.job_config[self.job_config_processing_inputs_key]

            for input in analytics_inputs:
                if input[self.job_config_input_name_key] == self.base_trial_input_channel_name:
                    s3_path = input[self.job_config_s3_input_key][self.job_config_s3_path]
                    break
        except Exception as e:
            print(f"Got exception while getting s3_input_path {e}")
        return s3_path


# output_html_filename doesn't require html extension. html extension will automatically be created.
def convert_notebook_to_html(
    notebook_full_path, output_html_full_dir, output_html_filename, config=None
):
    if config is None:
        c = Config()
        c.TagRemovePreprocessor.remove_input_tags = ("hide-input",)
        c.TagRemovePreprocessor.remove_cell_tags = ("hide-cell",)
        c.TagRemovePreprocessor.remove_all_outputs_tags = ("hide-output",)
        c.HTMLExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]
    else:
        c = config
    output, resource = HTMLExporter(config=c).from_filename(notebook_full_path)
    cc = Config()
    cc.FilesWriter.build_directory = output_html_full_dir
    fw = FilesWriter(config=cc)
    fw.write(output, resource, notebook_name=output_html_filename)


class PlotVisualizations(Rule):
    def __init__(
        self,
        base_trial=None,
        base_trial_path="",
        create_html="True",
        nb_full_path="/opt/ml/code/plot_viz_rule.ipynb",
        output_full_path="/opt/ml/processing/output/rule/plot_viz_rule.ipynb",
        processing_job_config_path="/opt/ml/config/processingjobconfig.json",
    ):
        """

        :param base_trial: the trial whose execution will invoke the rule

        """

        if base_trial is not None:
            self.base_trial_path = base_trial.path
        else:
            self.base_trial_path = base_trial_path
            self.base_trial = create_trial(self.base_trial_path)
        self._last_found_step = -1
        self._create_html = parse_bool(create_html, True)
        self._notebook_full_path = nb_full_path
        self._output_full_path = output_full_path
        self._processing_job_config_json_path = processing_job_config_path
        self._processing_config = ProcessingJobConfig(
            processing_job_config_path=self._processing_job_config_json_path
        )
        super().__init__(base_trial, action_str="")
        self.logger.info(
            f"Rule PlotVisualizations created with base_trial_path:{self.base_trial_path} run_jupyter_book:{self._create_html} nb_full_path:{nb_full_path} output_full_path:{output_full_path}"
        )

        ## TODO
        # Take scalar collections to be plotted as parameter
        # Take histogram regex

    def _plot_visualization(self, last_found_step):
        # TODO take save_steps as parameter,
        #  save_steps would make sure that rule is executed only every_save_steps
        output_dir = os.path.dirname(self._output_full_path)
        # putting in .sagemaker-ignore will make sure s3 agent doesn't upload the temp file
        output_notebook_path_tmp = os.path.join(output_dir, ".sagemaker-ignore", "out.tmp")
        ensure_dir(output_notebook_path_tmp)

        ## TODO check return type, try catch and let it retry for next step.
        s3_path = self._processing_config.get_s3_input_path()
        self.logger.info(f"Got s3_path:{s3_path}")
        if self.base_trial_path.startswith("s3://"):
            self.logger.info("Overriding s3 path to path:{self.base_trial_path}")
            s3_path = self.base_trial_path
        ret = pm.execute_notebook(
            self._notebook_full_path,
            output_notebook_path_tmp,
            # parameters=dict(path=self.base_trial_path, plot_step=last_found_step, s3_path=s3_path),
        )
        self.logger.info(f"notebook execute return code:{ret}")
        if os.path.exists(output_notebook_path_tmp):
            self.logger.info(f"Putting output notebook in {self._output_full_path}")
            os.rename(output_notebook_path_tmp, self._output_full_path)

        if self._create_html is True:
            html_dir_name = output_dir
            output_file_name = "profiler-report"
            self.logger.info(f"Putting html in {html_dir_name}/output_file_name.html")
            convert_notebook_to_html(
                notebook_full_path=self._output_full_path,
                output_html_full_dir=html_dir_name,
                output_html_filename=output_file_name,
            )

    def _complete_rule_if_all_steps_plotted(self, loaded_all_steps, steps, step):
        if loaded_all_steps:
            last_step = steps[-1] if len(steps) > 0 else -1
            raise NoMoreData(step, ModeKeys.GLOBAL, last_step)

    # returns 1 means thi step is processed
    # return 2 means this step need to be re-invoked
    # returns NoMoreData if this step is not going to come.
    def invoke_at_step(self, step):
        steps = self.base_trial.steps()
        self.logger.info(f"Invoked rule at step:{step}, steps found:{steps}")
        loaded_all_steps = self.base_trial.loaded_all_steps
        if len(steps) > 0 and steps[-1] >= step:
            # last step found is greater than this step, check if last step is not already processed
            if steps[-1] > self._last_found_step:
                self._last_found_step = steps[-1]
                self.logger.info(
                    f"New step {steps[-1]} found, updated last_step_found to {self._last_found_step}, plotting new data"
                )
                # new step found, plot new data
                self._plot_visualization(self._last_found_step)
            # last step found is already processed
            self._complete_rule_if_all_steps_plotted(loaded_all_steps, steps, step)
            return 1
        self._complete_rule_if_all_steps_plotted(loaded_all_steps, steps, step)
        # This means that we need to wait for this step as all step has not been loaded and
        return 2

    def invoke(self, step):
        val = 2
        # val can have 3 values - 0 indicates step step has been seen in training job, 1 indicates that new step
        # hasn't been seen for threshold time, 2 indicates that step is not available yet and we need to keep looking
        # for step
        while val == 2:  # step is not available yet
            val = self.invoke_at_step(step)
            time.sleep(1)
