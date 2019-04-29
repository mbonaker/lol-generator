import csv
import functools
import io
import re

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


class FieldSpecification:
    # Property Path,Format,Recommended Handling,Default,Occurrence,Mean,Standard Deviation,Mean Deviation,Min,Max,Median,Mode
    HANDLING_CONTINUOUS = 0
    HANDLING_ONEHOT = 1
    HANDLING_BOOL = 2
    HANDLING_NONE = 3
    HANDLING_UNARY = 4
    OCCURRENCE_SINGLE = 0
    OCCURRENCE_PERPARTICIPANT = 1
    OCCURRENCE_PERTEAM = 2

    @staticmethod
    def get_optional_field_names():
        for participant_id in range(0, 10):
            yield "participants.{:d}.championId".format(participant_id)
            for spell_id in (1, 2):
                yield "participants.{:d}.spell{:d}Id".format(participant_id, spell_id)
        for team_id in (0, 1):
            for ban_id in (0, 1, 2, 3, 4):
                yield "teams.{:d}.bans.{:d}.championId".format(team_id, ban_id)

    @staticmethod
    def from_dict(source: Dict[str, str], known_data_optional: bool = False):
        optional_column_names = list(FieldSpecification.get_optional_field_names())
        name = source["Property Path"]
        if source["Format"].startswith("enum("):
            levels = source["Format"][5:-1].split("|")
            if known_data_optional and name in optional_column_names:
                levels.append("")
            format_spec = np.array(levels)
            converter = str
        else:
            format_spec = {
                "int": int,
                "bool": bool,
                "float": float,
            }[source["Format"]]
            converter = format_spec
        handling = {
            "continuous": FieldSpecification.HANDLING_CONTINUOUS,
            "bool": FieldSpecification.HANDLING_BOOL,
            "one-hot": FieldSpecification.HANDLING_ONEHOT,
            "unary": FieldSpecification.HANDLING_UNARY,
        }[source["Recommended Handling"]]
        occurrence = {
            "single": FieldSpecification.OCCURRENCE_SINGLE,
            "per-participant": FieldSpecification.OCCURRENCE_PERPARTICIPANT,
            "per-team": FieldSpecification.OCCURRENCE_PERTEAM,
        }[source["Occurrence"]]
        mean = float(source["Mean"]) if source["Mean"] else None
        sd = float(source["Standard Deviation"]) if source["Standard Deviation"] else None
        md = float(source["Mean Deviation"]) if source["Mean Deviation"] else None
        min_value = converter(source["Min"]) if source["Min"] else None
        max_value = converter(source["Max"]) if source["Max"] else None
        median = converter(source["Median"]) if source["Median"] else None
        mode = converter(source["Mode"]) if source["Mode"] else None
        if name in optional_column_names:
            default = ""
        else:
            default_column = source["Default"]
            if default_column == "Mean":
                default = mean
            elif default_column == "Standard Deviation":
                default = sd
            elif default_column == "Mean Deviation":
                default = md
            else:
                default = converter(source[default_column])
        return FieldSpecification(name, format_spec, handling, default, occurrence, mean, sd, md, min_value, max_value, median, mode)

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

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, FieldSpecification):
            return False
        return self.name == other.name


class FieldStructure:
    @staticmethod
    @functools.lru_cache(maxsize=10)
    def make(data_path: str, portion: int = PORTION_INTERESTING, known_data_optional: bool = False):
        return FieldStructure(data_path, portion, known_data_optional)

    def __init__(self, data_path: str, portion: int = PORTION_INTERESTING, known_data_optional: bool = False):
        self.data_path = data_path
        self.logger = logging.getLogger(__name__)
        portion_set = set()
        if portion & PORTION_KNOWN:
            self.known_names = self.read_field_list_file("{path}/columns/known".format(path=data_path))
            portion_set.update(self.known_names)
        else:
            self.known_names = None
        if portion & PORTION_UNKNOWN - PORTION_WIN:
            self.unknown_names = self.read_field_list_file("{path}/columns/unknown".format(path=data_path))
            self.unknown_without_win_names = tuple(col for col in self.unknown_names if col not in ("teams.0.win", "teams.1.win"))
            portion_set.update(self.unknown_without_win_names)
        else:
            self.unknown_names = None
            self.unknown_without_win_names = None
        if (portion & PORTION_INTERESTING - PORTION_WIN) == PORTION_INTERESTING - PORTION_WIN:
            self.interesting_names = self.read_field_list_file("{path}/columns/interesting".format(path=data_path))
            self.interesting_without_win_names = tuple(col for col in self.interesting_names if col not in ("teams.0.win", "teams.1.win"))
            portion_set.update(self.interesting_without_win_names)
        else:
            self.interesting_names = None
            self.interesting_without_win_names = None
        self.win_names = ("teams.0.win", "teams.1.win")
        if portion & PORTION_WIN:
            portion_set.update(self.win_names)
        assert len(portion_set) > 0
        self.champion_names = self.read_names_csv("{path}/champion_names.csv".format(path=data_path))
        self.spell_names = self.read_names_csv("{path}/spell_names.csv".format(path=data_path))
        with open("{path}/columns/interesting.csv".format(path=data_path), "r") as f:
            self.specs: List[FieldSpecification] = list()
            csv_data = csv.DictReader(f)
            for line in csv_data:
                if line['Property Path'] in portion_set:
                    self.specs.append(FieldSpecification.from_dict(line, known_data_optional))

    @property
    def known(self) -> List[FieldSpecification]:
        return [field for field in self.specs if field.name in self.known_names]

    @property
    def unknown(self) -> List[FieldSpecification]:
        return [field for field in self.specs if field.name in self.unknown_names]

    @property
    def interesting(self) -> List[FieldSpecification]:
        return [field for field in self.specs if field.name in self.interesting_names]

    @property
    def unknown_without_win(self) -> List[FieldSpecification]:
        return [field for field in self.specs if field.name in self.unknown_without_win_names]

    @property
    def interesting_without_win(self) -> List[FieldSpecification]:
        return [field for field in self.specs if field.name in self.interesting_without_win_names]

    @property
    def win(self) -> List[FieldSpecification]:
        return [field for field in self.specs if field.name in self.win_names]

    @staticmethod
    def read_names_csv(full_path: str) -> Dict[int, str]:
        logger = logging.getLogger(__name__)
        logger.log(logging.DEBUG, "Reading {path!r}...".format(path=full_path))
        with open(full_path, "r") as f:
            csv_data = csv.DictReader(f, fieldnames=("id", "name"))
            return dict((int(line["id"]), line["name"]) for line in csv_data)

    @staticmethod
    def read_field_list_file(full_path: str) -> Tuple[str, ...]:
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
        return dict((col.name, col.dtype) for col in self.specs)

    @functools.lru_cache(maxsize=1024)
    def get_overall_min(self, field: FieldSpecification):
        if field.occurrence == FieldSpecification.OCCURRENCE_PERPARTICIPANT:
            subproperty_name = field.name[len("participants.X."):]
            all_property_names = tuple("participants.{pid}.{sp}".format(pid=pid, sp=subproperty_name) for pid in range(0, 9))
            return min(field.min_value for field in self.specs if field.name in all_property_names)
        if field.occurrence == FieldSpecification.OCCURRENCE_PERTEAM:
            subproperty_name = field.name[len("teams.X."):]
            all_property_names = tuple("teams.{tid}.{sp}".format(tid=tid, sp=subproperty_name) for tid in (0, 1))
            return min(field.min_value for field in self.specs if field.name in all_property_names)
        return field.min_value

    @functools.lru_cache(maxsize=1024)
    def get_overall_max(self, field: FieldSpecification):
        if field.occurrence == FieldSpecification.OCCURRENCE_PERPARTICIPANT:
            subproperty_name = field.name[len("participants.X."):]
            all_property_names = tuple("participants.{pid}.{sp}".format(pid=pid, sp=subproperty_name) for pid in range(0, 9))
            return max(field.max_value for field in self.specs if field.name in all_property_names)
        if field.occurrence == FieldSpecification.OCCURRENCE_PERTEAM:
            subproperty_name = field.name[len("teams.X."):]
            all_property_names = tuple("teams.{tid}.{sp}".format(tid=tid, sp=subproperty_name) for tid in (0, 1))
            return max(field.max_value for field in self.specs if field.name in all_property_names)
        return field.max_value


class ColumnSpecification:
    def __init__(self, field: FieldSpecification):
        self.field = field
        self.handling = field.handling

    @property
    def name(self):
        return self.field.name

    def __str__(self):
        return self.name


class OneHotFieldColumnSpecification(ColumnSpecification):
    def __init__(self, field: FieldSpecification, key: str):
        super().__init__(field)
        self.key = key

    @property
    def name(self):
        return "{column_name}.{one_hot_level}".format(column_name=self.field.name, one_hot_level=self.key)


class UnaryFieldColumnSpecification(ColumnSpecification):
    def __init__(self, field: FieldSpecification, number: int):
        super().__init__(field)
        self.number = number

    @property
    def name(self):
        return "{column_name}.{number:d}".format(column_name=self.field.name, number=self.number)


class ColumnStructure:
    @staticmethod
    @functools.lru_cache(maxsize=10)
    def make(field_structure: FieldStructure, dtype: np.dtype, portion: int = PORTION_INTERESTING):
        return ColumnStructure(field_structure, dtype, portion)

    def __init__(self, field_structure: FieldStructure, dtype: np.dtype, portion: int = PORTION_INTERESTING):
        # logging
        self.logger = logging.getLogger(__name__)
        logger = logging.getLogger(__name__)
        logger.log(logging.DEBUG, "Initializing field structure to column structure conversion...")

        # initialize variables
        self.portion = portion
        self.fields = field_structure
        self.dtype = dtype
        self.slices = dict()

        # create the column representations
        self.specs: List[ColumnSpecification] = []
        for field in field_structure.specs:
            # if it is not part of this corpus
            use_column = False
            if self.portion & PORTION_KNOWN and field in self.fields.known:
                use_column = True
            if self.portion & PORTION_WIN and field in self.fields.win:
                use_column = True
            if self.portion & (PORTION_UNKNOWN - PORTION_WIN) and field in self.fields.unknown_without_win:
                use_column = True
            if not use_column:
                continue

            # create the column representation(s) which are multiple in case of one-hot encoding
            if field.handling == FieldSpecification.HANDLING_BOOL:
                self.specs.append(ColumnSpecification(field))
            elif field.handling == FieldSpecification.HANDLING_CONTINUOUS:
                self.specs.append(ColumnSpecification(field))
            elif field.handling == FieldSpecification.HANDLING_ONEHOT:
                assert hasattr(field.format_spec, '__iter__') or field.format_spec == bool
                for key in (True, False) if field.format_spec is bool else field.format_spec:
                    self.specs.append(OneHotFieldColumnSpecification(field, key))
            elif field.handling == FieldSpecification.HANDLING_UNARY:
                overall_min = field_structure.get_overall_min(field)
                overall_max = field_structure.get_overall_max(field)
                for number in range(overall_min + 1, overall_max + 1):
                    self.specs.append(UnaryFieldColumnSpecification(field, number))
            else:
                raise ValueError("Column handling type unknown")

        # Sort the column representations so that 'win' are the last two and 'known' comes first.
        # This way we can use slicing and numpy views to access the different portions in most cases.
        column_order = []
        if self.portion & PORTION_KNOWN:
            column_order.extend(self.fields.known)
        if self.portion & (PORTION_UNKNOWN - PORTION_WIN):
            column_order.extend(u for u in self.fields.unknown if u not in self.fields.win)
        if self.portion & PORTION_WIN:
            column_order.extend(self.fields.win)
        self.specs.sort(key=lambda col: column_order.index(col.field))
        field = None
        field_start = None
        for i, column in enumerate(self.specs):
            if column.field != field:
                if field is not None:
                    self.slices[field.name] = slice(field_start, i)
                field = column.field
                field_start = i
        self.slices[self.specs[-1].field.name] = slice(field_start, None)

    def get_portion_slice(self, portion: int):
        if not self.portion & portion:
            raise LookupError("The portion to take a slice from is at least partially not contained in this corpus.")
        if portion == PORTION_KNOWN:
            for index, col in enumerate(self.specs):
                if col.field not in self.fields.known:
                    return slice(0, index)
            return slice(0, len(self.specs))
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
                end = -2
            else:
                end = None
            return slice(start, end)
        elif portion == PORTION_INTERESTING:
            return slice(0, None)
        elif portion == PORTION_INTERESTING - PORTION_WIN:
            interesting_slice = self.interesting_slice
            if self.portion & PORTION_WIN:
                end = -2
            else:
                end = None
            return slice(interesting_slice.start, end)
        elif portion == PORTION_WIN:
            return slice(-2, None)

    def get_fields_by_portion(self, portion: int) -> Iterable[FieldSpecification]:
        if portion == PORTION_KNOWN:
            return self.fields.known
        elif portion == PORTION_WIN:
            return self.fields.win
        elif portion == PORTION_UNKNOWN:
            return self.fields.unknown
        elif portion == PORTION_INTERESTING:
            return self.fields.interesting
        elif portion == PORTION_INTERESTING - PORTION_WIN:
            return self.fields.interesting_without_win
        elif portion == PORTION_UNKNOWN - PORTION_WIN:
            return self.fields.unknown_without_win
        else:
            raise ValueError("Unknown portion {:d}".format(portion))

    def get_portion_indices(self, portion: int) -> Generator[int, None, None]:
        if self.portion & portion != portion:
            raise LookupError("The portion to take is at least partially not contained in this corpus.")
        if portion == self.portion:
            yield from range(len(self.specs))
        fields = self.get_fields_by_portion(portion)
        for index, col in enumerate(self.specs):
            if col.field in fields:
                yield index

    @property
    def known_slice(self) -> slice:
        return self.get_portion_slice(PORTION_KNOWN)

    @property
    def known_indices(self) -> List[int]:
        return list(self.get_portion_indices(PORTION_KNOWN))

    @property
    def unknown_slice(self) -> slice:
        return self.get_portion_slice(PORTION_UNKNOWN)

    @property
    def unknown_indices(self) -> List[int]:
        return list(self.get_portion_indices(PORTION_UNKNOWN))

    @property
    def unknown_without_win_slice(self) -> slice:
        return self.get_portion_slice(PORTION_UNKNOWN - PORTION_WIN)

    @property
    def unknown_without_win_indices(self) -> List[int]:
        return list(self.get_portion_indices(PORTION_UNKNOWN - PORTION_WIN))

    @property
    def interesting_slice(self) -> slice:
        return self.get_portion_slice(PORTION_INTERESTING)

    @property
    def interesting_indices(self) -> List[int]:
        return list(self.get_portion_indices(PORTION_INTERESTING))

    @property
    def interesting_without_win_slice(self) -> slice:
        return self.get_portion_slice(PORTION_INTERESTING - PORTION_WIN)

    @property
    def interesting_without_win_indices(self) -> List[int]:
        return list(self.get_portion_indices(PORTION_INTERESTING - PORTION_WIN))

    @property
    def win_slice(self) -> slice:
        return self.get_portion_slice(PORTION_WIN)

    @property
    def win_indices(self) -> List[int]:
        return list(self.get_portion_indices(PORTION_WIN))

    def pd2np(self, dataframe: pandas.DataFrame, ndarray: np.ndarray) -> None:
        self.logger.log(logging.DEBUG, "Convert pandas dataframe to ndarray...")
        for i, column in enumerate(self.specs):
            field = column.field
            pd_col = dataframe[field.name]
            if isinstance(column, OneHotFieldColumnSpecification):
                ndarray[:, i] = (pd_col == column.key).astype(self.dtype)
            elif isinstance(column, UnaryFieldColumnSpecification):
                ndarray[:, i] = (pd_col >= column.number).astype(self.dtype)
            elif column.handling == FieldSpecification.HANDLING_BOOL and pd_col.dtype.name == "category":
                ndarray[:, i] = (pd_col == field.mode).astype(self.dtype)
            elif column.handling == FieldSpecification.HANDLING_BOOL and is_numeric_dtype(pd_col.dtype):
                ndarray[:, i] = (pd_col > field.mean).astype(self.dtype)
            else:
                mean = field.mean
                sd = field.sd
                ndarray[:, i] = (pd_col - mean) / sd

    def np2pd(self, dataframe: pandas.DataFrame, ndarray: np.ndarray) -> None:
        assert dataframe.shape[0] == ndarray.shape[0]
        fields = set(column.field for column in self.specs)
        for field in fields:
            column_slice = self.field_to_column_slice(field)
            assert column_slice.stop is None or column_slice.stop < ndarray.shape[1]
            if field.handling == FieldSpecification.HANDLING_ONEHOT:
                level_indices = np.argmax(ndarray[:, column_slice], axis=1)
                dataframe[field.name] = field.format_spec[level_indices]
            elif field.handling == FieldSpecification.HANDLING_UNARY:
                number = np.sum(ndarray[:, column_slice] > 0, 1) + field.min_value
                dataframe[field.name] = number
            elif field.handling == FieldSpecification.HANDLING_BOOL and isinstance(field.format_spec, Iterable):
                level_indices = (ndarray[:, column_slice] > 0).astype(np.int32)
                true_value = field.mode
                false_value = next(level for level in field.format_spec if level != true_value)
                dataframe[field.name] = np.array([false_value, true_value])[level_indices]
            elif field.handling == FieldSpecification.HANDLING_BOOL and (field.format_spec is int or field.format_spec is float):
                level_indices = (ndarray[:, column_slice] > 0).astype(np.int32)
                mean = field.mean
                sd = field.sd
                true_value = mean + sd
                false_value = mean - sd
                dataframe[field.name] = np.array([false_value, true_value])[level_indices]
            elif field.handling == FieldSpecification.HANDLING_BOOL and field.format_spec is bool:
                level_indices = (ndarray[:, column_slice] > 0).astype(np.int32)
                dataframe[field.name] = np.array([False, True])[level_indices]
            else:
                mean = field.mean
                sd = field.sd
                out = np.float64(ndarray[:, column_slice]) * sd + mean
                if field.min_value >= 0:
                    out = np.maximum(out, 0)
                if field.format_spec is float:
                    out = np.minimum(out, np.finfo(field.dtype).max)
                elif field.format_spec is int:
                    out = np.minimum(out, np.iinfo(field.dtype).max)
                dataframe[field.name] = out

    def shuffle_participants(self, data: np.ndarray, random_state: np.random.RandomState) -> np.ndarray:
        shuffled_data = np.ndarray(data.shape, data.dtype)
        shuffled_data[:, :] = data
        for tid in (0, 1):
            for cont_member_id, rand_member_id in enumerate(list(random_state.permutation(5))):
                if cont_member_id == rand_member_id:
                    continue
                cont_pid = tid * 5 + cont_member_id
                rand_pid = tid * 5 + rand_member_id
                cont_key_part = "participants.{pid:d}.".format(pid=cont_pid)
                cont_ban_key_part = "teams.{tid:d}.bans.{mid:d}.".format(tid=tid, mid=cont_member_id)
                rand_key_part = "participants.{pid:d}.".format(pid=rand_pid)
                rand_ban_key_part = "teams.{tid:d}.bans.{mid:d}.".format(tid=tid, mid=rand_member_id)
                cont_cids = np.where([col.name.startswith(cont_key_part) or col.name.startswith(cont_ban_key_part) for col in self.specs])[0]
                rand_cids = np.where([col.name.startswith(rand_key_part) or col.name.startswith(rand_ban_key_part) for col in self.specs])[0]
                shuffled_data[:, cont_cids] = data[:, rand_cids]
        return shuffled_data

    def column_indices_from_names(self, names: Iterable[str]) -> List[int]:
        indices = []
        for index, column in enumerate(self.specs):
            if column.field.name in names:
                indices.append(index)
        return indices

    def field_name_to_column_slice(self, name: str) -> slice:
        return self.slices[name]

    def field_to_column_slice(self, field: FieldSpecification) -> slice:
        return self.slices[field.name]

    def generate_slices(self, query: Callable[[FieldSpecification], bool], max_slices: Optional[int] = None) -> Generator[slice, None, None]:
        hit_start = None
        slice_amount = 0
        for i, column in enumerate(self.specs):
            field = column.field
            hit = query(field)
            if hit and hit_start is None:
                hit_start = i
            elif not hit and hit_start is not None:
                yield slice(hit_start, i)
                hit_start = None
                slice_amount += 1
                if slice_amount == max_slices:
                    return
        if hit_start is not None:
            yield slice(hit_start, None)

    def generate_handling_slices(self, ignored_columns: Iterable[str] = tuple()):
        current_handling = None
        current_start = 0
        current_end = 0
        for column in self.specs:
            field = column.field
            if field.name in ignored_columns:
                handling = FieldSpecification.HANDLING_NONE
            else:
                handling = field.handling
            if current_handling is not None and handling != current_handling:
                yield (slice(current_start, current_end), current_handling)
                current_start = current_end
                current_end += 1
                current_handling = handling
            else:
                current_end += 1
            if current_handling is None:
                current_handling = handling
        if current_handling is not None:
            yield (slice(current_start, current_end), current_handling)

    def randomly_unspecify_optional_columns(self, ndarray: np.ndarray, random_state: np.random.RandomState, p: float = 0.5, q: float = 0.1) -> np.ndarray:
        rows_kept = random_state.choice(ndarray.shape[0], int(ndarray.shape[0] * q))
        for field_name in FieldSpecification.get_optional_field_names():
            column_slice = self.field_name_to_column_slice(field_name)
            ndarray = self._randomly_unspecify_column(ndarray, column_slice, random_state, rows_kept, p)
        return ndarray

    def _randomly_unspecify_column(self, ndarray: np.ndarray, column_slice: slice, random_state: np.random.RandomState, rows_kept: np.ndarray, p: float = 0.5) -> np.ndarray:
        column_contents = slice(column_slice.start, column_slice.stop - 1)
        column_empty = column_slice.stop - 1
        mask = np.zeros(ndarray.shape[0], dtype=np.bool_)
        mask[random_state.choice(ndarray.shape[0], int(ndarray.shape[0] * p))] = True
        mask[rows_kept] = False
        ndarray[mask, column_contents] = 0
        ndarray[mask, column_empty] = 1
        return ndarray


class DataProvider:
    def __init__(self, data_path: str, dtype: np.dtype, portion: int = PORTION_INTERESTING, known_data_is_optional: bool = False):
        self.logger = logging.getLogger(__name__)
        self.data_path = data_path
        self.fields = FieldStructure.make(data_path, portion, known_data_is_optional)
        self.columns = ColumnStructure.make(self.fields, dtype, portion)
        self.data: Optional[np.ndarray] = None

    @property
    def known(self) -> np.ndarray:
        return self.data[:, self.columns.known_slice]

    @known.setter
    def known(self, value) -> None:
        self.data[:, self.columns.known_slice] = value

    @property
    def unknown(self) -> np.ndarray:
        return self.data[:, self.columns.unknown_slice]

    @unknown.setter
    def unknown(self, value) -> None:
        self.data[:, self.columns.unknown_slice] = value

    @property
    def unknown_without_win(self) -> np.ndarray:
        return self.data[:, self.columns.unknown_without_win_slice]

    @unknown_without_win.setter
    def unknown_without_win(self, value) -> None:
        self.data[:, self.columns.unknown_without_win_slice] = value

    @property
    def interesting(self) -> np.ndarray:
        return self.data[:, self.columns.interesting_slice]

    @interesting.setter
    def interesting(self, value) -> None:
        self.data[:, self.columns.interesting_slice] = value

    @property
    def interesting_without_win(self) -> np.ndarray:
        return self.data[:, self.columns.interesting_without_win_slice]

    @interesting_without_win.setter
    def interesting_without_win(self, value) -> None:
        self.data[:, self.columns.interesting_without_win_slice] = value

    @property
    def win(self) -> np.ndarray:
        return self.data[:, self.columns.win_slice]

    @win.setter
    def win(self, value) -> None:
        self.data[:, self.columns.win_slice] = value

    def get_ndarray(self, portion: Optional[int] = None):
        if portion is None:
            return self.data
        else:
            return self.data[:, self.columns.get_portion_slice(portion)]

    def get_as_dataframe(self, nd=None) -> pandas.DataFrame:
        nd = self.get_ndarray(self.columns.portion) if nd is None else nd
        self.logger.log(logging.DEBUG, "Calculating the dataframe for numpy data (np shape: {shape!r})...".format(shape=nd.shape))
        dataframe = pandas.DataFrame(index=pandas.RangeIndex(0, nd.shape[0]), columns=tuple(col.name for col in self.fields.specs))
        self.columns.np2pd(dataframe, nd)
        self.logger.log(logging.DEBUG, "Convert data frame parts to corresponding types...")
        for col_name, dtype in self.fields.dtype.items():
            try:
                dataframe[col_name] = dataframe[col_name].astype(dtype)
            except (TypeError, ValueError):
                # ignore, because we only offer type conversion as a convenience without guarantee
                pass
        self.logger.log(logging.DEBUG, "Return data frame...")
        return dataframe

    def write_as_csv(self, file: io.TextIOBase, nd=None):
        dataframe = self.get_as_dataframe(nd)
        for column in self.fields.specs:
            dataframe[column.name] = dataframe[column.name].fillna(column.default)
        self.logger.log(logging.DEBUG, "Write data frame as csv to stream...")
        dataframe.to_csv(file, header=False, index=False)

    def create_empty_data(self, length: int):
        self.data = np.ndarray(shape=(length, len(self.columns.specs)), dtype=self.columns.dtype)

    def create_nan_data(self, length: int):
        self.data = np.full(shape=(length, len(self.columns.specs)), fill_value=np.nan, dtype=self.columns.dtype)

    def get_nan_data(self, length: int) -> np.ndarray:
        return np.full(shape=(length, len(self.columns.specs)), fill_value=np.nan, dtype=self.columns.dtype)


class KnownStreamProvider(DataProvider):
    def __init__(self, stream, data_path: str, dtype: np.dtype, known_data_is_optional: bool = False):
        super().__init__(data_path, dtype, PORTION_KNOWN, known_data_is_optional)
        self.stream = stream
        if self.stream:
            try:
                self.load()
            except BaseException as e:
                self.logger.error("Could not load the necessary data from stdin")
                raise DataHandlingError("Could not load the necessary data from stdin") from e

    def load_get(self, stream) -> np.ndarray:
        self.logger.log(logging.INFO, "Reading known matches from stdin...")
        dataframe = pandas.read_csv(
            stream,
            dtype=self.fields.dtype,
            header=None,
            names=self.fields.known_names,
            true_values=("True",),
            na_values=('',),
            keep_default_na=False,
        )
        for field in self.fields.specs:
            if field.name in dataframe:
                dataframe[field.name] = dataframe[field.name].fillna(field.default).astype(field.dtype)
        ndarray: np.ndarray = np.ndarray(shape=(dataframe.shape[0], len(self.columns.specs)), dtype=self.columns.dtype)
        self.columns.pd2np(dataframe, ndarray)
        return ndarray

    def load(self) -> None:
        self.data = self.load_get(self.stream)


class CorpusProvider(DataProvider):
    def __init__(self, data_path: str, dtype: np.dtype, portion: int = PORTION_INTERESTING, known_data_is_optional: bool = False):
        super().__init__(data_path, dtype, portion, known_data_is_optional)
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
            self.logger.info("Memmap numpy array was created. Now delete all remakes...")
            duration = self.columns.field_name_to_column_slice("gameDuration")
            i = 0
            c = 0
            while self.data.shape[0] > i:
                self.data[c, :] = self.data[i, :]
                if self.data[i, duration] > (15 * 60 - 1762.503) / 493.163:
                    c += 1
                i += 1
            self.logger.info("Memmap numpy array was created. Now delete all remakes...")
            self.save_to_npy_file(c)

    def load_from_npy_file(self) -> np.ndarray:
        self.logger.log(logging.INFO, "Load {npy_path}".format(npy_path=self.npy_file_name))
        data = np.load(self.npy_file_name, mmap_mode="r")
        self.logger.log(logging.INFO, "Loaded {npy_path}".format(npy_path=self.npy_file_name))
        return data

    @property
    def npy_file_name(self) -> str:
        return "{path}/Matches.npy".format(path=self.data_path)

    def save_to_npy_file(self, match_amount=None) -> None:
        self.logger.log(logging.INFO, "Save {npy_path}".format(npy_path=self.npy_file_name))
        np.save(self.npy_file_name, self.data[:match_amount, :])
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
                for column in self.fields.specs:
                    chunk[column.name] = chunk[column.name].fillna(column.default).astype(column.dtype)
                self.columns.pd2np(chunk, memmap[i * chunksize:i * chunksize + chunk.shape[0], :])
                chunk_queue.task_done()
                self.logger.log(logging.INFO, "Thread {id:d} is done with conversion of chunk #{i:d}".format(id=thread_index, i=i))

    def load_from_csv_file(self) -> np.ndarray:
        self.logger.log(logging.INFO, "Importing Matches.csv...")
        self.logger.log(logging.INFO, "Calculating the amount of matches in Matches.csv to initialize matrix of appropriate size...")
        match_count = self.fields.count_matches()
        with tempfile.NamedTemporaryFile(dir=self.data_path) as f:
            ndarray: np.ndarray = np.memmap(f, mode="w+", shape=(match_count, len(self.columns.specs)), dtype=self.columns.dtype)
        chunksize = 1 << 13
        total_chunk_amount = math.ceil(match_count / chunksize)
        cpu_count = os.cpu_count() or 1
        thread_count = max(1, cpu_count - 1)
        threads = list()
        chunk_queue = queue.Queue(maxsize=thread_count)
        self.logger.log(logging.INFO, "Create {:d} threads...".format(thread_count))
        for i in range(thread_count):  # 'cpu_count - 1' because the main cpu is busy filling the queue
            new_thread = threading.Thread(target=CorpusProvider.dataframe_to_ndarray_conversion_thread, args=(self, chunk_queue, ndarray, chunksize, i))
            new_thread.start()
            threads.append(new_thread)
        self.logger.log(logging.INFO, "Begin reading Matches.csv...")
        for i, chunk in enumerate(pandas.read_csv(
            "{path}/Matches.csv".format(path=self.data_path),
            dtype=self.fields.dtype,
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
