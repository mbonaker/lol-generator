# dataprovider.py

The module _dataprovider_ can be used independently from the rest of this repository. It provides a convenient way to read the _Matches.csv_ file partially into memory, as needed.

## Setup

To use the _dataprovider_ module, you may need about 11 GB of free disk space at some point:
 * ~3.9 GB to store the _Matches.csv_ file
 * ~5 GB initially to store the temporary data while converting the _Matches.csv_ to a _Matches.npz_
 * ~1.3 GB to store the _Matches.npz_)
 
 You need to have one _data_ directory anywhere on the PC and in it there needs to be this file structure (basically just a `git clone` of the _data_ git repository **plus** the _Matches.csv_ file):
 
 * champion_names.csv
 * spell_names.csv
 * Matches.csv
 * columns
   * all
   * interesting
   * interesting.csv
   * known
   * uninteresting
   * unknown

## Usage

```python
import numpy as np
import dataprovider

# Assuming the _data_ repository was cloned to `C:\\Path\\to\\data\\`.
# get the python data matrix ready to go
data = dataprovider.CorpusProvider("C:\\Path\\to\\data\\", np.dtype(np.float32))

# get the 'known' data without 'win'
known_data = data.unknown_without_win

# only use the first half of that 'known - win' data
known_first_half = np.array_split(known_data, 2)[0]

# other available data partitions are these:
partitions = [
    data.known,
    data.unknown,
    data.unknown_without_win,
    data.interesting,
    data.interesting_without_win,
]

# let's print the shapes:
partition_names = ("known", "unknown", "unknown_without_win", "interesting", "interesting_without_win")
print("\n".join("{n} with shape {p.shape!r}".format(p=p, n=n) for p, n in zip(partitions, partition_names)))
```

# generator

Documentation not yet done...
