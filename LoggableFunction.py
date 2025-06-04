from typing import Callable, Generic, List, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class LoggableFunction(Generic[P, R]):
    """Abstract class for a function that includes important metadata to log for an experiment, such as weight
    generation or edge sampling. Inherit from this to use it in order to add more metadata (e.g. tau/alpha for GIRG
    generation) and override the type information for the function. Can be called like a normal function."""
    _function: Callable[P, R]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._function(*args, **kwargs)

    @property
    def function_role(self) -> str:
        """Returns a string describing what the function is being used for, e.g. "Initial vertex selector"."""
        raise NotImplementedError()

    @property
    def log_lines(self) -> List[str]:
        """Returns a list of lines to log all public members of the function. Allows for nested LoggableFunctions."""
        line_list = [f"{self.function_role} of type {type(self)}:\n"]

        for attr, value in self.__dict__.items():
            # Ignore private members.
            if attr[0] == "_":
                continue
            # If this isn't a LoggableFunction, just cast to string and append the name and value.
            if not issubclass(self.__class__, type(value)):
                line_list.append(f"\t{attr}=={value}\n")
            # Otherwise, we add a level of indentation and recurse into the member's log_lines function.
            else:
                line_list.append(f"\tMember {attr}:\n")
                added_lines = ['\t\t' + line for line in value.log_lines()]
                line_list.extend(added_lines)

        return line_list
