import csv
import tempfile
import os
import sys
import numpy as np
import pandas
import logging
from typing import *


from indexed import IndexedOrderedDict
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.api.types import is_numeric_dtype


class CsvColumnSpecification:
    # Property Path,Format,Recommended Handling,Default,Occurrence,Mean,Standard Deviation,Mean Deviation,Min,Max,Median,Mode
    HANDLING_CONTINUOUS = 0
    HANDLING_ONEHOT = 1
    HANDLING_BOOL = 2
    OCCURRENCE_SINGLE = 0
    OCCURRENCE_PERPARTICIPANT = 1
    OCCURRENCE_PERTEAM = 2
    @staticmethod
    def from_dict(source: Dict[str, str]):
        name = source["Property Path"]
        if source["Format"].startswith("enum("):
            format_spec = np.array(source["Format"][5:-1].split("|"))
            converter = str
        else:
            format_spec = {
                "int": int,
                "bool": bool,
                "float": float,
            }[source["Format"]]
            converter = format_spec
        handling = {
            "continuous": CsvColumnSpecification.HANDLING_CONTINUOUS,
            "bool": CsvColumnSpecification.HANDLING_BOOL,
            "one-hot": CsvColumnSpecification.HANDLING_ONEHOT,
        }[source["Recommended Handling"]]
        occurrence = {
            "single": CsvColumnSpecification.OCCURRENCE_SINGLE,
            "per-participant": CsvColumnSpecification.OCCURRENCE_PERPARTICIPANT,
            "per-team": CsvColumnSpecification.OCCURRENCE_PERTEAM,
        }[source["Occurrence"]]
        mean = float(source["Mean"]) if source["Mean"] else None
        sd = float(source["Standard Deviation"]) if source["Standard Deviation"] else None
        md = float(source["Mean Deviation"]) if source["Mean Deviation"] else None
        min_value = converter(source["Min"]) if source["Min"] else None
        max_value = converter(source["Max"]) if source["Max"] else None
        median = converter(source["Median"]) if source["Median"] else None
        mode = converter(source["Mode"]) if source["Mode"] else None
        default_column = source["Default"]
        if default_column == "Mean":
            default = mean
        elif default_column == "Standard Deviation":
            default = sd
        elif default_column == "Mean Deviation":
            default = md
        else:
            default = converter(source[default_column])
        return CsvColumnSpecification(name, format_spec, handling, default, occurrence, mean, sd, md, min_value, max_value, median, mode)

    def __init__(self, name: str, format_spec, handling: int, default, occurrence: int, mean: Optional[float], sd: Optional[float], md: Optional[float], min_value, max_value, median, mode):
        self.name = name
        self.format_spec = format_spec
        self.handling = handling
        self.default = default
        self.occurrence = occurrence
        self.mean = mean
        self.sd = sd
        self.md = md
        self.min_value = min_value
        self.max_value = max_value
        self.median = median
        self.mode = mode

        self.dtype: Union[np.dtype, pandas.api.types.CategoricalDtype]
        if isinstance(self.format_spec, np.ndarray) or isinstance(self.format_spec, list):
            self.dtype = CategoricalDtype(categories=self.format_spec, ordered=False)
        elif self.format_spec == int:
            unsigned = self.min_value >= 0
            extra_bits = 0 if unsigned else 1
            extreme_value = max(abs(self.max_value), abs(self.min_value))
            if extreme_value >> 8 * 1 - extra_bits == 0:
                byte_size = 1
            elif extreme_value >> 8 * 2 - extra_bits == 0:
                byte_size = 2
            elif extreme_value >> 8 * 4 - extra_bits == 0:
                byte_size = 4
            elif extreme_value >> 8 * 8 - extra_bits == 0:
                byte_size = 8
            else:
                raise ValueError("{!r} contains too large numbers for numpy".format(self.name))
            self.dtype = np.dtype("{type}{byte_size}".format(type="u" if unsigned else "i", byte_size=byte_size))
        elif self.format_spec == float:
            self.dtype = np.dtype("f4")
        elif self.format_spec == bool:
            self.dtype = np.dtype("?")
        else:
            raise ValueError("format_spec {!r} unknown".format(self.format_spec))


class CsvCorpusStructure:
    def __init__(self, data_path: str):
        self.logger = logging.getLogger(__name__)
        self.all = self.read_column_list_file("{path}/columns/all".format(path=data_path))
        self.interesting = self.read_column_list_file("{path}/columns/interesting".format(path=data_path))
        self.known = self.read_column_list_file("{path}/columns/known".format(path=data_path))
        self.unknown = self.read_column_list_file("{path}/columns/unknown".format(path=data_path))
        self.uninteresting = self.read_column_list_file("{path}/columns/uninteresting".format(path=data_path))
        self.champion_names = self.read_names_csv("{path}/champion_names.csv".format(path=data_path))
        self.spell_names = self.read_names_csv("{path}/spell_names.csv".format(path=data_path))
        with open("{path}/columns/interesting.csv".format(path=data_path), "r") as f:
            self.columns: MutableMapping[str, CsvColumnSpecification] = IndexedOrderedDict()
            csv_data = csv.DictReader(f)
            for line in csv_data:
                spec = CsvColumnSpecification.from_dict(line)
                self.columns[spec.name] = spec

    @staticmethod
    def read_names_csv(full_path: str) -> Dict[int, str]:
        logger = logging.getLogger(__name__)
        logger.log(logging.DEBUG, "Reading {path!r}...".format(path=full_path))
        with open(full_path, "r") as f:
            csv_data = csv.DictReader(f, fieldnames=("id", "name"))
            return dict((int(line["id"]), line["name"]) for line in csv_data)

    @staticmethod
    def read_column_list_file(full_path: str) -> Tuple[str, ...]:
        logger = logging.getLogger(__name__)
        logger.log(logging.DEBUG, "Reading {path!r}...".format(path=full_path))
        with open(full_path, "r") as f:
            return tuple(l if l[-1] != "\n" else l[:-1] for l in f)

    @property
    def dtype(self) -> Dict[str, Union[np.dtype, pandas.api.types.CategoricalDtype]]:
        return dict((col.name, col.dtype) for col in self.columns.values())

    @property
    def dtype_known(self) -> Dict[str, Union[np.dtype, pandas.api.types.CategoricalDtype]]:
        return dict((col.name, col.dtype) for col in self.columns.values() if col.name in self.known)


class NumpyColumnSpecification:
    def __init__(self, csv_column_specification: CsvColumnSpecification):
        self.csv_column_specification = csv_column_specification
        self.handling = csv_column_specification.handling

    @property
    def name(self):
        return self.csv_column_specification.name


class NumpyOneHotFieldColumnSpecification(NumpyColumnSpecification):
    def __init__(self, csv_column_specification: CsvColumnSpecification, key: str):
        super().__init__(csv_column_specification)
        self.key = key

    @property
    def name(self):
        return "{column_name}.{one_hot_level}".format(column_name=self.csv_column_specification.name, one_hot_level=self.key)


class NumpyCorpusStructure:
    PORTION_KNOWN = 1
    PORTION_WIN = 2
    PORTION_UNKNOWN = 2 + 4
    PORTION_INTERESTING = 1 + 2 + 4 + 8

    def __init__(self, csv_structure: CsvCorpusStructure, dtype: np.dtype, portion: Optional[int] = None):
        # logging
        self.logger = logging.getLogger(__name__)
        logger = logging.getLogger(__name__)
        logger.log(logging.DEBUG, "Initializing csv structure to numpy structure conversion...")

        # initialize variables
        self.portion = portion if portion is not None else self.PORTION_INTERESTING
        self.known = csv_structure.known
        self.unknown = csv_structure.unknown
        self.win = ('teams.0.win', 'teams.1.win')
        self.interesting = csv_structure.interesting
        self.dtype = dtype

        # create the column representations
        self.columns: List[NumpyColumnSpecification] = []
        for csv_column in csv_structure.columns.values():
            # if it is not part of this corpus
            if csv_column in self.known and not self.portion & self.PORTION_KNOWN:
                continue
            if csv_column in self.win and not self.portion & self.PORTION_WIN:
                continue
            if csv_column in self.unknown and not self.portion & (self.PORTION_UNKNOWN - self.PORTION_WIN):
                continue

            # create the column representation(s) which are multiple in case of one-hot encoding
            if csv_column.handling == CsvColumnSpecification.HANDLING_BOOL:
                self.columns.append(NumpyColumnSpecification(csv_column))
            elif csv_column.handling == CsvColumnSpecification.HANDLING_CONTINUOUS:
                self.columns.append(NumpyColumnSpecification(csv_column))
            elif csv_column.handling == CsvColumnSpecification.HANDLING_ONEHOT:
                assert hasattr(csv_column.format_spec, '__iter__') or csv_column.format_spec == bool
                for key in (True, False) if csv_column.format_spec is bool else csv_column.format_spec:
                    self.columns.append(NumpyOneHotFieldColumnSpecification(csv_column, key))
            else:
                raise ValueError("Column handling type unknown")

        # Sort the column representations so that 'win' are the last two and 'known' comes first.
        # This way we can use slicing and numpy views to access the different portions in most cases.
        column_order = []
        if self.portion & self.PORTION_KNOWN:
            column_order.extend(self.known)
        if self.portion & (self.PORTION_UNKNOWN - self.PORTION_WIN):
            column_order.extend(u for u in self.unknown if u not in self.win)
        if self.portion & self.PORTION_WIN:
            column_order.extend(self.win)
        self.columns.sort(key=lambda col: column_order.index(col.csv_column_specification.name))

    @property
    def known_slice(self) -> slice:
        if not self.portion & self.PORTION_KNOWN:
            raise LookupError("The 'known' portion is not contained in this corpus.")
        index = 0
        for index, col in enumerate(self.columns):
            if col.csv_column_specification.name not in self.known:
                return slice(0, index - 1)
        return slice(0, index)

    @property
    def unknown_slice(self) -> slice:
        if not self.portion & self.PORTION_UNKNOWN:
            raise LookupError("The 'unknown' portion is not contained in this corpus.")
        if not self.portion & self.PORTION_KNOWN:
            return slice(0, None)
        index = self.known_slice.stop
        return slice(index, None)

    @property
    def unknown_without_win_slice(self) -> slice:
        if not self.portion & (self.PORTION_UNKNOWN - self.PORTION_WIN):
            raise LookupError("The 'unknown' portion is not contained in this corpus.")
        if self.portion & self.PORTION_KNOWN:
            start = self.unknown_slice.start
        else:
            start = 0
        if self.portion & self.PORTION_WIN:
            end = -2
        else:
            end = None
        return slice(start, end)

    @property
    def interesting_slice(self) -> slice:
        if not self.portion & self.PORTION_INTERESTING:
            raise LookupError("The 'interesting' portion is not contained in this corpus.")
        return slice(0, None)

    @property
    def interesting_without_win_slice(self) -> slice:
        if not self.portion & (self.PORTION_INTERESTING - self.PORTION_WIN):
            raise LookupError("The 'interesting' portion is not contained in this corpus.")
        interesting_slice = self.interesting_slice
        if self.portion & self.PORTION_WIN:
            end = -2
        else:
            end = None
        return slice(interesting_slice.start, end)

    @property
    def win_slice(self) -> slice:
        if not self.portion & self.PORTION_WIN:
            raise LookupError("The 'win' portion is not contained in this corpus.")
        return slice(-2, None)

    def csv_structured_to_np_structured(self, data: pandas.DataFrame) -> np.ndarray:
        self.logger.log(logging.DEBUG, "Converting actual pandas.DataFrame to numpy.ndarray...")
        with tempfile.NamedTemporaryFile() as f:
            new_data = np.memmap(f, mode="a+", shape=(data.shape[0], len(self.columns)), dtype=self.dtype)
        for i, np_col_spec in enumerate(self.columns):
            csv_col_spec = np_col_spec.csv_column_specification
            pd_col = data[csv_col_spec.name]
            self.logger.log(logging.DEBUG, "Converting column {col!r} from pandas.DataFrame to np.ndarray format".format(col=csv_col_spec.name))
            if isinstance(np_col_spec, NumpyOneHotFieldColumnSpecification):
                new_data[:, i] = (pd_col == np_col_spec.key).astype(self.dtype)
            elif np_col_spec.handling == CsvColumnSpecification.HANDLING_BOOL and pd_col.dtype.name == "category":
                new_data[:, i] = (pd_col == csv_col_spec.mode).astype(self.dtype)
            elif np_col_spec.handling == CsvColumnSpecification.HANDLING_BOOL and is_numeric_dtype(pd_col.dtype):
                new_data[:, i] = (pd_col > csv_col_spec.mean).astype(self.dtype)
            else:
                mean = csv_col_spec.mean
                sd = csv_col_spec.sd
                new_data[:, i] = (pd_col - mean) / sd
        self.logger.log(logging.DEBUG, "pandas.DataFrame to numpy.ndarray conversion done.")
        return new_data

    def column_indices_from_names(self, names: Iterable[str]) -> List[int]:
        indices = []
        for index, np_col_spec in enumerate(self.columns):
            if np_col_spec.csv_column_specification.name in names:
                indices.append(index)
        return indices


class DataProvider:
    def __init__(self, data_path: str, dtype: np.dtype):
        self.logger = logging.getLogger(__name__)
        self.data_path = data_path
        self.csv_structure = CsvCorpusStructure(data_path)
        self.np_structure = NumpyCorpusStructure(self.csv_structure, dtype)
        self.data: np.ndarray = None

    @property
    def known(self):
        return self.data[:, self.np_structure.known_slice]

    @property
    def unknown(self):
        return self.data[:, self.np_structure.unknown_slice]

    @property
    def unknown_without_win(self):
        return self.data[:, self.np_structure.unknown_without_win_slice]

    @property
    def interesting(self):
        return self.data[:, self.np_structure.interesting_slice]

    @property
    def interesting_without_win(self):
        return self.data[:, self.np_structure.interesting_without_win_slice]

    @property
    def win(self):
        return self.data[:, self.np_structure.win_slice]


class KnownStdinProvider:
    def __init__(self, data_path: str, dtype: np.dtype):
        self.logger = logging.getLogger(__name__)
        self.data_path = data_path
        self.csv_structure = CsvCorpusStructure(data_path)
        self.np_structure = NumpyCorpusStructure(self.csv_structure, dtype)
        self.data: np.ndarray = None

    def load_from_csv_stdin(self):
        self.logger.log(logging.INFO, "Reading known matches from stdin...")
        data = pandas.read_csv(
            sys.stdin,
            dtype=self.csv_structure.dtype,
            header=0,
            true_values=("True",),
            na_values=('',),
            keep_default_na=False,
        )
        for column in self.csv_structure.columns.values():
            data[column.name] = data[column.name].fillna(column.default).astype(column.dtype)
        self.data = self.np_structure.csv_structured_to_np_structured(data)


class CorpusProvider(DataProvider):
    def __init__(self, data_path: str, dtype: np.dtype):
        super().__init__(data_path, dtype)
        self.logger = logging.getLogger(__name__)
        self.load()

    def load(self):
        if os.path.isfile(self.npy_file_name):
            self.load_from_npy_file()
        else:
            self.load_from_csv_file()
            self.save_to_npy_file()

    def load_from_npy_file(self):
        self.logger.log(logging.INFO, "Load {npy_path}".format(npy_path=self.npy_file_name))
        self.data = np.load(self.npy_file_name, mmap_mode="r")
        self.logger.log(logging.INFO, "Loaded {npy_path}".format(npy_path=self.npy_file_name))

    @property
    def npy_file_name(self):
        return "{path}/Matches.npy".format(path=self.data_path)

    def save_to_npy_file(self):
        self.logger.log(logging.INFO, "Save {npy_path}".format(npy_path=self.npy_file_name))
        np.save(self.npy_file_name, self.data)
        self.logger.log(logging.INFO, "Saved {npy_path}".format(npy_path=self.npy_file_name))

    def load_from_csv_file(self):
        self.logger.log(logging.INFO, "Importing Matches.csv...")
        data = pandas.read_csv(
            "{path}/Matches.csv".format(path=self.data_path),
            dtype=self.csv_structure.dtype,
            header=0,
            true_values=("True",),
            na_values=('',),
            keep_default_na=False,
        )
        for column in self.csv_structure.columns.values():
            data[column.name] = data[column.name].fillna(column.default).astype(column.dtype)
        self.data = self.np_structure.csv_structured_to_np_structured(data)
