from SIEpidemic import SIEpidemic
from typing import Any,  List, Optional, Union
from pathlib import Path
import numpy as np
import dill  # type: ignore
from LoggableFunction import LoggableFunction


class InitialVertexFunction(LoggableFunction[[np.random.Generator], int]):
    """Function to choose the initial vertex of each run of an experiment. Includes a name for logging."""
    pass


class GenericInitialVertexFunction(InitialVertexFunction):
    """Lightweight option to just pass in the function you care about with a description for logging."""
    def __init__(self, _function, description: str) -> None:
        self._function = _function
        self.description = description


class ResultFunction(LoggableFunction[[SIEpidemic, np.random.Generator], Any]):
    """Function to pull out the results of each run of an experiment. Includes a name for logging."""
    pass


class GenericResultFunction(ResultFunction):
    """Lightweight option to just pass in the function you care about with a description for logging."""
    def __init__(self, _function, description: str) -> None:
        self._function = _function
        self.description = description


class SIExperiment:
    """
    Class to run and log an experiment with multiple iterations of an SI epidemic. Note that optional arguments are
    only optional in order to allow loading them from a file.

    Attributes:
        epidemic: The SIEpidemic itself, which contains all information about the graph and how to resample it.
        run_count: The number of SI epidemics to simulate.
        resample_edges: Whether the edges of the graph should be resampled between runs.
        resample_costs: Whether the edge costs of the graph should be resampled between runs.
        initial_vertex_fn: A function to determine the (possibly random) initially-infected vertex ID each run.
        log_path: The folder to log experiment data to.
        name: All experiment data will appear in files starting with [name] in log_path, e.g. [name].seed.
        full_log: Logs the entire SIEpidemic each run if true. Otherwise only logs the results. These will be logged to
            log_path, but with -run-[number] appended to the end of the file.
        result_fn: A function to pull the (possibly random) parameters of interest each run from the SIEpidemic.
        seed: The RNG seed for the current iteration.
        reset_seed: True if the seed should be reset (for logging purposes) between successive runs of the experiment.
            Defaults to False if a seed is specified (in which case only the RNG is re-initialised), True otherwise.
        generator: The numpy RNG to use for all random events.
        results: The results of the experiment, stored as a numbered list.
    """
    def __init__(self, epidemic: Optional[SIEpidemic], run_count: int, resample_edges: bool, resample_costs: bool,
                 initial_vertex_fn: Optional[InitialVertexFunction], log_path: Path, name: str, full_log: bool,
                 result_fn: Optional[ResultFunction], seed: Optional[int] = None) -> None:
        self.epidemic = epidemic
        self.run_count = run_count
        self.resample_edges = resample_edges
        self.resample_costs = resample_costs
        self.initial_vertex_fn = initial_vertex_fn
        self.log_path = log_path
        self.name = name
        self.full_log = full_log
        self.result_fn = result_fn
        self.seed = seed if seed else np.random.SeedSequence().entropy
        self.reset_seed = True if seed else False
        self.generator = np.random.default_rng(self.seed)
        self.results: List[Any] = []
        self._current_run = 0

    def _single_run(self, resample_edges: bool) -> None:
        """Executes a single run of the experiment, resampling and/or logging the SIEpidemic if if necessary."""
        if self.epidemic is None:
            raise RuntimeError("Attempting to run experiment with an undefined epidemic.")
        if self.initial_vertex_fn is None:
            raise RuntimeError("Attempting to run experiment with an undefined epidemic.")
        if self.result_fn is None:
            raise RuntimeError("Attempting to run experiment with an undefined epidemic.")

        if resample_edges:
            self.epidemic.sample_edges()
        if self.resample_costs:
            self.epidemic.sample_edge_costs()
        self.epidemic.run_infection(initial_vertex_id=self.initial_vertex_fn(self.generator))
        self.results.append(self.result_fn(self.epidemic.graph, self.generator))
        if self.full_log:
            self._log_run()

    def execute(self) -> List[Any]:
        """Carries out the entire experiment, logging everything necessary to file and self.results (and also returning
        it for convenience). Resets the seed and generator afterwards if reset_seed is True."""
        self.results = []
        self._current_run = 0

        self._log_settings()

        for i in range(self.run_count):
            # We don't bother resampling immediately before the first run, as we sampled once on class creation.
            self._single_run(resample_edges=self.resample_edges and i != 0)
            self._current_run += 1
        self._log_results()

        if self.reset_seed:
            self.seed = np.random.SeedSequence().entropy
            self.generator = np.random.default_rng(self.seed)

        return self.results

    @property
    def brief_logs_exist(self) -> bool:
        """Returns true if a seed and results log for this experiment already exists."""
        results_exist = (self.log_path / self.results_name).exists()
        settings_exist = (self.log_path / self.settings_name).exists()
        functions_exist = (self.log_path / self.functions_name).exists()
        return results_exist and settings_exist and functions_exist

    @property
    def full_logs_exist(self) -> bool:
        """Returns true if full logs for this experiment already exist, including the seed, results and SIEpidemics."""
        for i in range(self.run_count):
            if not (self.log_path / self.run_name(i)).exists():
                return False
        return self.brief_logs_exist

    def run_name(self, run_number: int) -> str:
        """Returns the filename prefix to save or load a given run number."""
        return f"{self.name}-run-{run_number}"

    def _log_run(self) -> None:
        """Saves the current SIEpidemic (via its own method)."""
        if self.epidemic is None:
            raise RuntimeError("Attempting to save a non-existent run.")
        self.epidemic.save_to_file(folder=self.log_path, name=self.run_name(self._current_run))

    def load_run(self, run_number: int) -> Optional[SIEpidemic]:
        """Returns the SIEpidemic from the given run index loaded from file."""
        return SIEpidemic.load_from_file(folder=self.log_path, name=self.run_name(run_number))

    @property
    def results_name(self) -> str:
        """Returns the filename of the results log."""
        return f"{self.name}-results.pickle"

    def _log_results(self) -> None:
        """Pickles the results of all SIEpidemic runs to self.log_path."""
        with open(self.log_path / self.results_name, "wb") as file:
            dill.dump(self.results, file)

    def load_results(self) -> List[Any]:
        """Returns the current set of results loaded from file."""
        with open(self.log_path / self.results_name, "rb") as file:
            return dill.load(file)

    @property
    def settings_name(self) -> str:
        """Returns the filename used to log the SIExperiment's settings."""
        return f"{self.name}-settings.cfg"

    def _log_settings(self) -> None:
        """Saves the current settings in human-readable format."""
        if self.initial_vertex_fn is None:
            raise RuntimeError("Attempting to save a non-existent initial vertex function.")
        if self.result_fn is None:
            raise RuntimeError("Attempting to save a non-existent result extraction function.")

        # Start with all the simple values that we can load directly.
        lines = [
            f"name=={self.name}\n",
            f"log_path=={self.log_path}\n",
            f"full_log=={self.full_log}\n",
            f"run_count=={self.run_count}\n",
            f"resample_edges=={self.resample_edges}\n",
            f"resample_costs=={self.resample_costs}\n",
            f"seed=={self.seed}\n",
            f"reset_seed=={self.reset_seed}\n"
        ]
        if self.epidemic is not None:
            lines.extend([f"mu=={self.epidemic.mu}\n", f"zeta=={self.epidemic.zeta}\n",
                          f"dimension=={self.epidemic.vertex_set.dimension}\n",
                          f"vertex_description=={self.epidemic.vertex_set.description}\n"])

        # The rest is text logging only, the actual functions are saved separately in pickled form.
        lines.extend(self.initial_vertex_fn.log_lines)
        lines.extend(self.result_fn.log_lines)
        if self.epidemic is not None:
            lines.extend(self.epidemic.edge_cost_generator.log_lines)
            lines.extend(self.epidemic.edge_generator.log_lines)
            lines.extend(self.epidemic.vertex_set.weight_generator.log_lines)
            lines.extend(self.epidemic.vertex_set.metric.log_lines)

        byte_lines = [line.encode("utf-8") for line in lines]
        with open(self.log_path / self.settings_name, "wb") as file:
            file.writelines(byte_lines)

    def load_settings(self) -> None:
        """Loads the settings of the last run from file into the current SIExperiment."""
        def check_valid(line_to_check: str, expected_lhs: str):
            if len(line_to_check.split("=")) != 2 or line_to_check.split("=")[0] != expected_lhs:
                raise ValueError(f"Bad settings file (line {line_to_check})")

        def extract_value(line_to_parse: str) -> Union[bool, int, str]:
            rhs = line_to_parse.split("==")[1]
            if rhs == "True":
                return True
            if rhs == "False":
                return False
            if rhs.isnumeric():
                return int(rhs)
            return rhs

        with open(self.log_path / self.settings_name, "rb") as file:
            fields = ["name", "log_path", "full_log"]
            for field in fields:
                line = file.readline().decode("utf-8")
                check_valid(line, field)
                setattr(self, field, extract_value(line))

    @property
    def functions_name(self) -> str:
        """Returns the filename of the functions for initial vertex choice and result extraction."""
        return f"{self.name}-functions.pickle"

    def _log_functions(self) -> None:
        """Pickles the functions for initial vertex choice and result extraction."""
        with open(self.log_path / self.results_name, "wb") as file:
            dill.dump(self.initial_vertex_fn, file)
            dill.dump(self.result_fn, file)

    def load_functions(self) -> None:
        """Loads the functions for initial vertex choice and result extraction to the current SIExperiment."""
        with open(self.log_path / self.results_name, "rb") as file:
            self.initial_vertex_fn = dill.load(file)
            self.result_fn = dill.load(file)

    @classmethod
    def LoadFromFile(cls, name: str, log_path: Path) -> "SIExperiment":
        """Loads the SIExperiment with the given name from the given folder. """
        return_value = SIExperiment(epidemic=None, run_count=0, resample_edges=False, resample_costs=False,
                                    log_path=log_path, name=name, full_log=False, initial_vertex_fn=None,
                                    result_fn=None)
        if not return_value.brief_logs_exist:
            raise FileNotFoundError(f"No existing logs found for the given SIExperiment '{name}'.")

        return_value.load_settings()
        return_value.load_functions()
        return_value.results = return_value.load_results()
        if return_value.full_logs_exist:
            return_value.epidemic = return_value.load_run(return_value.run_count - 1)
        else:
            print(f"WARNING: No full run logs exist for the given SIExperiment '{name}', epidemic set to None.")

        return return_value
