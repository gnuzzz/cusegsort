# cusegsort
Fast segmented sort on NVIDIA GPUs.

Based on <https://github.com/vtsynergy/bb_segsort>

## Usage:
* segmented key-value pairs sort: bb::segments::kv::bb_segsort
* segmented keys sort: bb::segments::k::bb_segsort
* sort matrix of key-value pairs by rows: bb::matrix::kv::bb_segsort
* sort matrix of keys by rows: bb::matrix::k::bb_segsort
* reuse internal buffers with bb::k::SortContext and bb::kv::SortContext

see examples in main.cu

## License: 
Please refer to the included LICENSE file.