import csv
import io
import math
import queue
import tempfile
import os
import sys
import threading

import numpy as np
import pandas
import logging
from typing import *


from indexed import IndexedOrderedDict
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.api.types import is_numeric_dtype

PORTION_KNOWN = 1
PORTION_WIN = 2
PORTION_UNKNOWN = 2 + 4
PORTION_INTERESTING = 1 + 2 + 4 + 8


class DataHandlingError(BaseException):
    pass


class FormattingError(BaseException):
    pass


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

    def __hash__(self):
        return hash(self.name)


class CsvCorpusStructure:
    def __init__(self, data_path: str, portion: int = PORTION_INTERESTING):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)
        portion_set = set()
        if portion & PORTION_KNOWN:
            self.known = self.read_column_list_file("{path}/columns/known".format(path=data_path))
            portion_set.update(self.known)
        else:
            self.known = None
        if portion & PORTION_UNKNOWN - PORTION_WIN:
            self.unknown = self.read_column_list_file("{path}/columns/unknown".format(path=data_path))
            self.unknown_without_win = tuple(col for col in self.unknown if col not in ("teams.0.win", "teams.1.win"))
            portion_set.update(self.unknown_without_win)
        else:
            self.unknown = None
            self.unknown_without_win = None
        if portion & PORTION_INTERESTING - PORTION_WIN:
            self.interesting = self.read_column_list_file("{path}/columns/interesting".format(path=data_path))
            self.interesting_without_win = tuple(col for col in self.interesting if col not in ("teams.0.win", "teams.1.win"))
            portion_set.update(self.interesting_without_win)
        else:
            self.interesting = None
            self.interesting_without_win = None
        if portion & PORTION_WIN:
            portion_set.update(("teams.0.win", "teams.1.win"))
        assert len(portion_set) > 0
        self.champion_names = self.read_names_csv("{path}/champion_names.csv".format(path=data_path))
        self.spell_names = self.read_names_csv("{path}/spell_names.csv".format(path=data_path))
        with open("{path}/columns/interesting.csv".format(path=data_path), "r") as f:
            self.columns: MutableMapping[str, CsvColumnSpecification] = IndexedOrderedDict()
            csv_data = csv.DictReader(f)
            for line in csv_data:
                if line['Property Path'] in portion_set:
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

    def count_matches(self):
        with open("{path}/Matches.csv".format(path=self.data_path), "r") as f:
            lines = 0
            buf_size = 1 << 20
            read_f = f.read  # loop optimization

            buf = read_f(buf_size)
            while buf:
                lines += buf.count('\n')
                buf = read_f(buf_size)
        # subtract 1 because the header doesn't count
        return lines - 1

    @property
    def dtype(self) -> Dict[str, Union[np.dtype, pandas.api.types.CategoricalDtype]]:
        return dict((col.name, col.dtype) for col in self.columns.values())



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

    def __init__(self, csv_structure: CsvCorpusStructure, dtype: np.dtype, portion: int = PORTION_INTERESTING):
        # logging
        self.logger = logging.getLogger(__name__)
        logger = logging.getLogger(__name__)
        logger.log(logging.DEBUG, "Initializing csv structure to numpy structure conversion...")

        # initialize variables
        self.portion = portion
        self.known = csv_structure.known
        self.unknown = csv_structure.unknown
        self.unknown_without_win = csv_structure.unknown_without_win
        self.win = ('teams.0.win', 'teams.1.win')
        self.interesting = csv_structure.interesting
        self.interesting_without_win = csv_structure.interesting_without_win
        self.dtype = dtype

        # create the column representations
        self.columns: List[NumpyColumnSpecification] = []
        for csv_column in csv_structure.columns.values():
            # if it is not part of this corpus
            use_column = False
            if self.portion & PORTION_KNOWN and csv_column.name in self.known:
                use_column = True
            if self.portion & PORTION_WIN and csv_column.name in self.win:
                use_column = True
            if self.portion & (PORTION_UNKNOWN - PORTION_WIN) and csv_column.name in self.unknown_without_win:
                use_column = True
            if not use_column:
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
        if self.portion & PORTION_KNOWN:
            column_order.extend(self.known)
        if self.portion & (PORTION_UNKNOWN - PORTION_WIN):
            column_order.extend(u for u in self.unknown if u not in self.win)
        if self.portion & PORTION_WIN:
            column_order.extend(self.win)
        self.columns.sort(key=lambda col: column_order.index(col.csv_column_specification.name))

    def get_portion_slice(self, portion: int):
        if not self.portion & portion:
            raise LookupError("The portion to take a slice from is not contained in this corpus.")
        if portion == PORTION_KNOWN:
            index = 0
            for index, col in enumerate(self.columns):
                if col.csv_column_specification.name not in self.known:
                    return slice(0, index - 1)
            return slice(0, index)
        elif portion == PORTION_UNKNOWN:
            if not self.portion & PORTION_KNOWN:
                return slice(0, None)
            index = self.known_slice.stop
            return slice(index, None)
        elif portion == PORTION_UNKNOWN - PORTION_WIN:
            if self.portion & PORTION_KNOWN:
                start = self.unknown_slice.start
            else:
                start = 0
            if self.portion & PORTION_WIN:
                end = -4
            else:
                end = None
            return slice(start, end)
        elif portion == PORTION_INTERESTING:
            return slice(0, None)
        elif portion == PORTION_INTERESTING - PORTION_WIN:
            interesting_slice = self.interesting_slice
            if self.portion & PORTION_WIN:
                end = -4
            else:
                end = None
            return slice(interesting_slice.start, end)
        elif portion == PORTION_WIN:
            return slice(-4, None)

    @property
    def known_slice(self) -> slice:
        return self.get_portion_slice(PORTION_KNOWN)

    @property
    def unknown_slice(self) -> slice:
        return self.get_portion_slice(PORTION_UNKNOWN)

    @property
    def unknown_without_win_slice(self) -> slice:
        return self.get_portion_slice(PORTION_UNKNOWN - PORTION_WIN)

    @property
    def interesting_slice(self) -> slice:
        return self.get_portion_slice(PORTION_INTERESTING)

    @property
    def interesting_without_win_slice(self) -> slice:
        return self.get_portion_slice(PORTION_INTERESTING - PORTION_WIN)

    @property
    def win_slice(self) -> slice:
        return self.get_portion_slice(PORTION_WIN)

    def pd2np(self, dataframe: pandas.DataFrame, ndarray: np.ndarray) -> None:
        for i, np_col_spec in enumerate(self.columns):
            csv_col_spec = np_col_spec.csv_column_specification
            pd_col = dataframe[csv_col_spec.name]
            if isinstance(np_col_spec, NumpyOneHotFieldColumnSpecification):
                ndarray[:, i] = (pd_col == np_col_spec.key).astype(self.dtype)
            elif np_col_spec.handling == CsvColumnSpecification.HANDLING_BOOL and pd_col.dtype.name == "category":
                ndarray[:, i] = (pd_col == csv_col_spec.mode).astype(self.dtype)
            elif np_col_spec.handling == CsvColumnSpecification.HANDLING_BOOL and is_numeric_dtype(pd_col.dtype):
                ndarray[:, i] = (pd_col > csv_col_spec.mean).astype(self.dtype)
            else:
                mean = csv_col_spec.mean
                sd = csv_col_spec.sd
                ndarray[:, i] = (pd_col - mean) / sd

    def np2pd(self, dataframe: pandas.DataFrame, ndarray: np.ndarray) -> None:
        assert dataframe.shape[0] == ndarray.shape[0]
        csv_col_specs = set(np_col_spec.csv_column_specification for np_col_spec in self.columns)
        for csv_col_spec in csv_col_specs:
            column_index = next(i for i, np_col_spec in enumerate(self.columns) if np_col_spec.csv_column_specification is csv_col_spec)
            if csv_col_spec.handling == CsvColumnSpecification.HANDLING_ONEHOT:
                start_column_index = column_index
                end_column_index = start_column_index + len(csv_col_spec.format_spec)
                level_indices = np.argmax(ndarray[:, start_column_index:end_column_index], axis=1)
                dataframe[csv_col_spec.name] = csv_col_spec.format_spec[level_indices]
            elif csv_col_spec.handling == CsvColumnSpecification.HANDLING_BOOL and isinstance(csv_col_spec.format_spec, Iterable):
                start_column_index = column_index
                end_column_index = start_column_index + 2
                level_indices = np.argmax(ndarray[:, start_column_index:end_column_index], axis=1)
                true_value = csv_col_spec.mode
                false_value = next(level for level in csv_col_spec.format_spec if level != true_value)
                dataframe[csv_col_spec.name] = np.array([false_value, true_value])[level_indices]
            elif csv_col_spec.handling == CsvColumnSpecification.HANDLING_BOOL and csv_col_spec.format_spec is int or csv_col_spec.format_spec is float:
                level_indices = (ndarray[:, column_index] > 0.5).astype(np.int32)
                mean = csv_col_spec.mean
                md = csv_col_spec.md
                true_value = mean + md
                false_value = mean - md
                dataframe[csv_col_spec.name] = np.array([false_value, true_value])[level_indices]
            else:
                mean = csv_col_spec.mean
                sd = csv_col_spec.sd
                dataframe[csv_col_spec.name] = ndarray[:, column_index] * sd + mean

    def column_indices_from_names(self, names: Iterable[str]) -> List[int]:
        indices = []
        for index, np_col_spec in enumerate(self.columns):
            if np_col_spec.csv_column_specification.name in names:
                indices.append(index)
        return indices


class DataProvider:
    def __init__(self, data_path: str, dtype: np.dtype, portion: int = PORTION_INTERESTING):
        self.logger = logging.getLogger(__name__)
        self.data_path = data_path
        self.csv_structure = CsvCorpusStructure(data_path, portion)
        self.np_structure = NumpyCorpusStructure(self.csv_structure, dtype, portion)
        self.data: np.ndarray = None

    @property
    def known(self) -> np.ndarray:
        return self.data[:, self.np_structure.known_slice]

    @known.setter
    def known(self, value) -> None:
        self.data[:, self.np_structure.known_slice] = value

    @property
    def unknown(self) -> np.ndarray:
        return self.data[:, self.np_structure.unknown_slice]

    @unknown.setter
    def unknown(self, value) -> None:
        self.data[:, self.np_structure.unknown_slice] = value

    @property
    def unknown_without_win(self) -> np.ndarray:
        return self.data[:, self.np_structure.unknown_without_win_slice]

    @unknown_without_win.setter
    def unknown_without_win(self, value) -> None:
        self.data[:, self.np_structure.unknown_without_win_slice] = value

    @property
    def interesting(self) -> np.ndarray:
        return self.data[:, self.np_structure.interesting_slice]

    @interesting.setter
    def interesting(self, value) -> None:
        self.data[:, self.np_structure.interesting_slice] = value

    @property
    def interesting_without_win(self) -> np.ndarray:
        return self.data[:, self.np_structure.interesting_without_win_slice]

    @interesting_without_win.setter
    def interesting_without_win(self, value) -> None:
        self.data[:, self.np_structure.interesting_without_win_slice] = value

    @property
    def win(self) -> np.ndarray:
        return self.data[:, self.np_structure.win_slice]

    @win.setter
    def win(self, value) -> None:
        self.data[:, self.np_structure.win_slice] = value

    def get_ndarray(self, portion: Optional[int] = None):
        if portion is None:
            return self.data
        else:
            return self.data[:, self.np_structure.get_portion_slice(portion)]

    def get_as_dataframe(self) -> pandas.DataFrame:
        self.logger.log(logging.DEBUG, "Calculating the dataframe for numpy data (np shape: {shape!r})...".format(shape=self.data.shape))
        dataframe = pandas.DataFrame(index=pandas.RangeIndex(0, self.data.shape[0]), columns=tuple(col.name for col in self.csv_structure.columns.values()))
        self.np_structure.np2pd(dataframe, self.data)
        for col_name, dtype in self.csv_structure.dtype.items():
            try:
                dataframe[col_name] = dataframe[col_name].astype(dtype)
            except (TypeError, ValueError):
                # ignore, because we only offer type conversion as a convenience without guarantee
                pass
        return dataframe

    def write_as_csv(self, file: io.TextIOBase):
        dataframe = self.get_as_dataframe()
        for column in self.csv_structure.columns.values():
            dataframe[column.name] = dataframe[column.name].fillna(column.default)
        dataframe.to_csv(file, header=False, index=False)

    def create_empty_data(self, length: int):
        self.data = np.ndarray(shape=(length, len(self.np_structure.columns)), dtype=self.np_structure.dtype)

    def create_nan_data(self, length: int):
        self.data = np.full(shape=(length, len(self.np_structure.columns)), fill_value=np.nan, dtype=self.np_structure.dtype)


class KnownStdinProvider(DataProvider):
    def __init__(self, data_path: str, dtype: np.dtype):
        super().__init__(data_path, dtype, PORTION_KNOWN)
        try:
            self.load()
        except BaseException as e:
            self.logger.error("Could not load the necessary data from stdin")
            raise DataHandlingError("Could not load the necessary data from stdin") from e

    def load(self) -> None:
        self.logger.log(logging.INFO, "Reading known matches from stdin...")
        dataframe = pandas.read_csv(
            sys.stdin,
            dtype=self.csv_structure.dtype,
            header=None,
            names=self.csv_structure.known,
            true_values=("True",),
            na_values=('',),
            keep_default_na=False,
        )
        for column in self.csv_structure.columns.values():
            if column.name in dataframe:
                dataframe[column.name] = dataframe[column.name].fillna(column.default).astype(column.dtype)
        ndarray: np.ndarray = np.ndarray(shape=(dataframe.shape[0], len(self.np_structure.columns)), dtype=self.np_structure.dtype)
        self.np_structure.pd2np(dataframe, ndarray)
        self.data = ndarray


class CorpusProvider(DataProvider):
    def __init__(self, data_path: str, dtype: np.dtype, portion: int = PORTION_INTERESTING):
        super().__init__(data_path, dtype, portion)
        try:
            self.load()
        except BaseException as e:
            self.logger.error("Could not load the necessary data for the corpus provider")
            raise DataHandlingError("Could not load the necessary data for the corpus provider") from e

    def load(self) -> None:
        if os.path.isfile(self.npy_file_name):
            self.data = self.load_from_npy_file()
        else:
            self.data = self.load_from_csv_file()
            self.save_to_npy_file()

    def load_from_npy_file(self) -> np.ndarray:
        self.logger.log(logging.INFO, "Load {npy_path}".format(npy_path=self.npy_file_name))
        data = np.load(self.npy_file_name, mmap_mode="r")
        self.logger.log(logging.INFO, "Loaded {npy_path}".format(npy_path=self.npy_file_name))
        return data

    @property
    def npy_file_name(self) -> str:
        return "{path}/Matches.npy".format(path=self.data_path)

    def save_to_npy_file(self) -> None:
        self.logger.log(logging.INFO, "Save {npy_path}".format(npy_path=self.npy_file_name))
        np.save(self.npy_file_name, self.data)
        self.logger.log(logging.INFO, "Saved {npy_path}".format(npy_path=self.npy_file_name))

    def dataframe_to_ndarray_conversion_thread(self, chunk_queue: queue.Queue, memmap: np.ndarray, chunksize: int, thread_index: int) -> None:
        done = False
        while not done:
            i, chunk = chunk_queue.get(block=True, timeout=None)
            if (i, chunk) == (None, None):
                done = True
                chunk_queue.task_done()
            else:
                self.logger.log(logging.DEBUG, "Start converting data #{i:d} on thread {id:d}".format(id=thread_index, i=i))
                for column in self.csv_structure.columns.values():
                    chunk[column.name] = chunk[column.name].fillna(column.default).astype(column.dtype)
                self.np_structure.pd2np(chunk, memmap[i * chunksize:i * chunksize + chunk.shape[0], :])
                chunk_queue.task_done()
                self.logger.log(logging.INFO, "Thread {id:d} is done with conversion of chunk #{i:d}".format(id=thread_index, i=i))

    def load_from_csv_file(self) -> np.ndarray:
        self.logger.log(logging.INFO, "Importing Matches.csv...")
        self.logger.log(logging.INFO, "Calculating the amount of matches in Matches.csv to initialize matrix of appropriate size...")
        match_count = self.csv_structure.count_matches()
        with tempfile.NamedTemporaryFile(dir=self.data_path) as f:
            ndarray: np.ndarray = np.memmap(f, mode="w+", shape=(match_count, len(self.np_structure.columns)), dtype=self.np_structure.dtype)
        chunksize = 1 << 13
        total_chunk_amount = math.ceil(match_count / chunksize)
        cpu_count = os.cpu_count() or 1
        thread_count = max(1, cpu_count - 1)
        threads = list()
        chunk_queue = queue.Queue(maxsize=thread_count)
        for i in range(thread_count):  # 'cpu_count - 1' because the main cpu is busy filling the queue
            new_thread = threading.Thread(target=CorpusProvider.dataframe_to_ndarray_conversion_thread, args=(self, chunk_queue, ndarray, chunksize, i))
            new_thread.start()
            threads.append(new_thread)
        for i, chunk in enumerate(pandas.read_csv(
            "{path}/Matches.csv".format(path=self.data_path),
            dtype=self.csv_structure.dtype,
            header=0,
            true_values=("True",),
            na_values=('',),
            keep_default_na=False,
            chunksize=chunksize,
        )):
            self.logger.log(logging.INFO, "Chunk {current_chunk_index:d} / {total_chunk_amount:d} read from CSV. Forward to conversion thread...".format(current_chunk_index=i + 1, total_chunk_amount=total_chunk_amount))
            chunk_queue.put((i, chunk), block=True, timeout=None)
        for i in range(len(threads)):
            chunk_queue.put((None, None), block=True, timeout=None)
        chunk_queue.join()
        return ndarray
