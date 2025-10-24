from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import dill  # type: ignore

from LoggableFunction import LoggableFunction
from SIEpidemic import SIEpidemic, EdgeGen, EdgeCostGen
from VertexSet import VertexSetGen
from WeightedVertexSet import WeightedVertexSet, WeightGen


class InitialVertexFunction(LoggableFunction[[SIEpidemic, np.random.Generator], int]):
    """Function to choose the ID of the initial vertex of each run of an experiment. Includes a name for logging."""
    @property
    def function_role(self):
        return "Initial vertex selector"


class FixedInitialVertex(InitialVertexFunction):
    def __init__(self, name: Any):
        self._function = lambda epidemic, _: epidemic.vertex_set.name_to_id(name)


class GenericInitialVertexFunction(InitialVertexFunction):
    """Lightweight option to just pass in the function you care about with a description for logging."""
    def __init__(self, _function, description: str) -> None:
        self._function = _function
        self.description = description


class ResultFunction(LoggableFunction[[SIEpidemic, np.random.Generator], Any]):
    """Function to pull out the results of each run of an experiment. Includes a name for logging."""
    @property
    def function_role(self):
        return "Test result extractor"


class GenericResultFunction(ResultFunction):
    """Lightweight option to just pass in the function you care about with a description for logging."""
    def __init__(self, _function, description: str) -> None:
        self._function = _function
        self.description = description


class SIExperiment:
    r"""
    Class to run and log an experiment with multiple iterations of an SI epidemic. Note that optional arguments are
    only optional in order to allow loading them from a file.

    Attributes:
        vertex_gen: The function to generate vertex sets for graphs.
        weight_gen: The function to generate weights for vertices.
        edge_gen: The function to generate edge sets for graphs.
        cost_gen: The function to generate costs for edges.
        current_run: The SIEpidemic from the current run, which contains all information about the graph.
        run_count: The number of SI epidemics to simulate.
        resample_costs: Whether the edge costs of the graph should be resampled between runs.
        resample_edges: Whether the entire edge set (including costs) should be resampled between runs.
        resample_weights: Whether the vertex weights should be resampled between runs.
        resample_vertices: Whether the entire vertex set (including weights) should be resampled between runs.
        initial_vertex_fn: A function to determine the (possibly random) initially-infected vertex ID each run.
        log_path: The folder to log experiment data to.
        name: All experiment data will appear in files starting with [name] in log_path, e.g. [name].seed.
        full_log: Logs the entire SIEpidemic each run if true. Otherwise only logs the results. These will be logged to
            log_path, but with -run-[number] appended to the end of the file.
        result_fn: A function to pull the (possibly random) parameters of interest each run from the SIEpidemic.
        seed: The RNG seed for the current iteration.
        rng: The numpy RNG to use for all random events.
        results: The results of the experiment, stored as a numbered list.
        mu: The weight penalty to apply to edge costs. (The cost of an edge {u,v} will be multiplied by
            (weight(u)*weight(v))^mu.)
        zeta: The distance penalty to apply to edge costs. (The cost of an edge {u,v} will be multiplied by
            ||u-v||^\zeta, where the distance function is determined by vertex_gen.)
    """
    def __init__(self, vertex_gen: Optional[VertexSetGen], weight_gen: Optional[WeightGen], edge_gen: Optional[EdgeGen],
                 cost_gen: Optional[EdgeCostGen], initial_vertex_fn: Optional[InitialVertexFunction],
                 result_fn: Optional[ResultFunction], run_count: int, resample_edges: bool, resample_costs: bool,
                 resample_weights: bool, resample_vertices: bool, log_path: Path, name: str, full_log: bool,
                 mu: float, zeta: float, seed: Optional[int] = None) -> None:
        self.seed = seed if seed else np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(self.seed)
        self.vertex_gen = vertex_gen
        self.weight_gen = weight_gen
        self.edge_gen = edge_gen
        self.cost_gen = cost_gen
        self.current_run: Optional[SIEpidemic] = None
        self.run_count = run_count
        self.resample_edges = resample_edges
        self.resample_costs = resample_costs
        self.resample_weights = resample_weights
        self.resample_vertices = resample_vertices
        self.initial_vertex_fn = initial_vertex_fn
        self.log_path = log_path
        self.name = name
        self.full_log = full_log
        self.result_fn = result_fn
        self.results: List[Any] = []
        self._current_run_count = 0
        self.mu = mu
        self.zeta = zeta

    def _single_run(self, first_run: bool) -> None:
        """Executes a single run of the experiment, resampling and/or logging the SIEpidemic if if necessary."""
        if self.current_run is None:
            raise RuntimeError("Attempting to run experiment with an undefined epidemic.")
        if self.initial_vertex_fn is None:
            raise RuntimeError("Attempting to run experiment with an undefined epidemic.")
        if self.result_fn is None:
            raise RuntimeError("Attempting to run experiment with an undefined epidemic.")

        if not first_run:
            if self.resample_vertices:
                self.current_run.vertex_set.resample_vertices(self.rng)
                self.current_run.sample_edges(self.rng)
                self.current_run.sample_edge_costs(self.rng)
            else:
                if self.resample_weights:
                    self.current_run.vertex_set.resample_weights(self.rng)
                if self.resample_edges:
                    self.current_run.sample_edges(self.rng)
                    self.current_run.sample_edge_costs(self.rng)
                elif self.resample_costs:
                    self.current_run.sample_edge_costs(self.rng)

        initial_vertex_id = self.initial_vertex_fn(self.current_run, self.rng)
        self.current_run.run_infection(initial_vertex_id=initial_vertex_id)
        self.results.append(self.result_fn(self.current_run, self.rng))
        if self.full_log:
            self._log_run()

    def execute(self) -> List[Any]:
        """Carries out the entire experiment, logging everything necessary to file and self.results (and also returning
        it for convenience)."""
        self.results = []
        self._current_run_count = 0

        self.save_config()

        if not (self.vertex_gen and self.weight_gen and self.edge_gen and self.cost_gen):
            raise RuntimeError("Attempting to run experiment with undefined generator functions.")
        vertices = WeightedVertexSet(vertex_gen=self.vertex_gen, weight_gen=self.weight_gen, rng=self.rng)
        self.current_run = SIEpidemic(vertex_set=vertices, cost_gen=self.cost_gen, rng=self.rng,
                                      edge_gen=self.edge_gen, mu=self.mu, zeta=self.zeta, name=self.name)

        if self.full_log and not (self.resample_vertices or self.resample_weights):
            self.current_run.save_vertices(self.log_path, run_index=None)
        for i in range(self.run_count):
            print(f"Evaluating infection {i+1}/{self.run_count}...")
            # We don't bother resampling immediately before the first run, as we sampled once on initialisation.
            self._single_run(first_run=(i == 0))
            self._current_run_count += 1

        print("Saving results...")
        self._log_results()

        return self.results

    @property
    def config_exists(self) -> bool:
        """Returns true if configuration information for this experiment has been logged."""
        settings_exist = (self.log_path / self.settings_name).exists()
        functions_exist = (self.log_path / self.functions_name).exists()
        return settings_exist and functions_exist

    @property
    def results_exist(self) -> bool:
        """Returns true if results for this experiment have been logged."""
        return (self.log_path / self.results_name).exists()

    @property
    def full_run_logs_exist(self) -> bool:
        """Returns true if full logs for every run of this experiment have been logged."""
        if not self.current_run:
            raise RuntimeError("Can only check for run logs when the epidemic has been defined.")
        for i in range(self.run_count):
            if not (self.log_path / self.current_run.graph_filename(i)).exists():
                return False
            if self.resample_vertices and not (self.log_path / self.current_run.vertex_filename(i)).exists():
                return False
        if not self.resample_vertices and not (self.log_path / self.current_run.vertex_filename(None)).exists():
            return False
        return True

    def _log_run(self) -> None:
        """Saves the current SIEpidemic (via its own method)."""
        if self.current_run is None:
            raise RuntimeError("Attempting to save a non-existent run.")
        self.current_run.save_graph(folder=self.log_path, run_index=self._current_run_count)
        self.current_run.save_config(folder=self.log_path)
        if self.resample_vertices or self.resample_weights:
            self.current_run.save_vertices(self.log_path, run_index=self._current_run_count)

    def load_run(self, run_number: int) -> SIEpidemic:
        """Loads the SIEpidemic from the given run index."""
        return SIEpidemic.load_full(folder=self.log_path, name=self.name, run_index=run_number)

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
        return f"{self.name}-experiment-settings.cfg"

    def save_config(self) -> None:
        """Saves the experiment configuration to file, to be reloaded later."""
        self._log_settings()
        self._log_functions()

    def _log_settings(self) -> None:
        """Saves the current settings in human-readable format."""
        if not (self.initial_vertex_fn and self.result_fn and self.vertex_gen and self.weight_gen and self.edge_gen
                and self.cost_gen):
            raise RuntimeError("Attempting to save settings, but not all settings are defined.")

        # Start with all the simple values that we can load directly.
        lines = [
            f"name=={self.name}\n",
            f"log_path=={self.log_path}\n",
            f"full_log=={self.full_log}\n",
            f"run_count=={self.run_count}\n",
            f"resample_edges=={self.resample_edges}\n",
            f"resample_costs=={self.resample_costs}\n",
            f"resample_weights=={self.resample_weights}\n",
            f"resample_vertices=={self.resample_vertices}\n",
            f"seed=={self.seed}\n",
            f"mu=={self.mu}\n",
            f"zeta=={self.zeta}\n"
        ]

        # The rest is text logging only, the actual information is saved separately in pickled form.
        lines.extend(self.initial_vertex_fn.log_lines)
        lines.extend(self.result_fn.log_lines)
        lines.extend(self.vertex_gen.log_lines)
        lines.extend(self.weight_gen.log_lines)
        lines.extend(self.edge_gen.log_lines)
        lines.extend(self.cost_gen.log_lines)
        if self.current_run is not None:
            lines.extend(self.current_run.vertex_set.metric.log_lines)

        byte_lines = [line.encode("utf-8") for line in lines]
        self.log_path.mkdir(exist_ok=True, parents=True)
        with open(self.log_path / self.settings_name, "wb") as file:
            file.writelines(byte_lines)

    def load_settings(self) -> None:
        """Loads the settings of the last run from file into the current SIExperiment."""
        def check_valid(line_to_check: str, expected_lhs: str):
            if len(line_to_check.split("==")) != 2 or line_to_check.split("==")[0] != expected_lhs:
                raise ValueError(f"Bad settings file (line {line_to_check})")

        def extract_value(line_to_parse: str) -> bool | float | int | str | Path:
            lhs = line_to_parse.split("==")[0]
            rhs = line_to_parse.split("==")[1][:-1]  # Remove trailing newline
            if rhs == "True":
                return True
            if rhs == "False":
                return False
            if rhs.isdecimal():
                return int(rhs)
            if lhs == "log_path":
                return Path(rhs)
            try:  # Yes, this really is the simplest way to do it.
                return float(rhs)
            except ValueError:
                return rhs

        with open(self.log_path / self.settings_name, "rb") as file:
            fields = ["name", "log_path", "full_log", "run_count", "resample_edges", "resample_costs",
                      "resample_weights", "resample_vertices", "seed", "mu", "zeta"]
            for field in fields:
                line = file.readline().decode("utf-8")
                check_valid(line, field)
                setattr(self, field, extract_value(line))

        self.rng = np.random.default_rng(self.seed)

    @property
    def functions_name(self) -> str:
        """Returns the filename of the functions for initial vertex choice and result extraction."""
        return f"{self.name}-experiment-functions.pickle"

    def _log_functions(self) -> None:
        """Pickles the functions for initial vertex choice and result extraction."""
        with open(self.log_path / self.functions_name, "wb") as file:
            dill.dump(self.vertex_gen, file)
            dill.dump(self.weight_gen, file)
            dill.dump(self.edge_gen, file)
            dill.dump(self.cost_gen, file)
            dill.dump(self.initial_vertex_fn, file)
            dill.dump(self.result_fn, file)

    def load_functions(self) -> None:
        """Loads the functions for initial vertex choice and result extraction to the current SIExperiment."""
        with open(self.log_path / self.functions_name, "rb") as file:
            self.vertex_gen = dill.load(file)
            self.weight_gen = dill.load(file)
            self.edge_gen = dill.load(file)
            self.cost_gen = dill.load(file)
            self.initial_vertex_fn = dill.load(file)
            self.result_fn = dill.load(file)

    @classmethod
    def load_from_file(cls, name: str, folder: Path, rerun: bool, new_name: Optional[str] = None,
                       use_old_seed: bool = True) -> "SIExperiment":
        """Loads the SIExperiment with the given name from the given folder. If rerun is true, loads from configuration
        and initialises ready to rerun from scratch; otherwise, loads the results and an existing saved graph. If
        rerun is true and new_name is set, uses the new name for the new experiment (to avoid overwriting old results);
        otherwise, uses the original name."""

        return_value = SIExperiment(vertex_gen=None, weight_gen=None, edge_gen=None, cost_gen=None, run_count=0,
                                    resample_edges=False, resample_costs=False, resample_vertices=False,
                                    resample_weights=False, log_path=folder, name=name, full_log=False,
                                    initial_vertex_fn=None, result_fn=None, mu=0., zeta=0.)

        if not return_value.config_exists:
            raise FileNotFoundError(f"No existing logs found for the given SIExperiment '{name}'.")
        return_value.load_settings()
        if not use_old_seed:
            return_value.seed = np.random.SeedSequence().entropy
            return_value.rng = np.random.default_rng(return_value.seed)
        return_value.load_functions()

        if rerun and new_name:
            return_value.name = new_name
            return return_value

        if not return_value.results_exist:
            raise FileNotFoundError(f"No results found for the given SIExperiment '{name}'.")

        return_value.results = return_value.load_results()
        return_value.current_run = return_value.load_run(return_value.run_count - 1)
        if not return_value.full_run_logs_exist:
            raise FileNotFoundError(f"Run logs missing for the given SIExperiment '{name}'.")

        return return_value
