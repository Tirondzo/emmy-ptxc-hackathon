"""Hacks, in the Codeforces sense: problems from tests/problems we propose for the hidden set.

Each is an emmy-generated folder in the open set's format (f32 only, as the hosted runner takes),
inside the five categories, and exercises something no open problem does. The reason says what
a compiler that passes the open set can still get wrong. `pytest tests -m hack` runs them.
emmy's own `.ref.cu` fails the p9x ones: its CUDA renders `//` and `%` as C's, which agrees with
torch only for non-negative operands; expected.npz is torch's answer.
"""

HACKS = {  # problem id -> what it catches
    "f40_softmax_16": "rows of 16 over 128 threads: 112 threads combine empty online-softmax states, and a max "
                      "seeded at -inf gives exp(-inf - -inf) = NaN; emmy seeds -1e30",
    "a22_keys_1000": "the only 16-thread block and WarpShuffle of length 16: a 0xffffffff member mask names lanes "
                     "that do not exist, and a fixed 16-8-4-2-1 butterfly reads them",
    "r46_sum_5": "5 elements over 128 threads: the strided loop runs zero times for most threads (a do-while "
                 "reads past the end), which then carry the identity through the shuffle and the tree",
    "p25_3d": "`//` in an index expression, which no open problem has: a pointwise tile of rank >= 2 decodes "
              "its flat index as (i // w, i % w)",
    "r23_unpinned_sum": "a sum split over 64 blocks: float atomics into a scalar output (`sum_1[] +=`) that the "
                        "runner zeroes; the open set has atomics only in a matmul",
    "p64_pow3": "pow, absent from the open set, of a negative base: `ex2(y * lg2(x))` is NaN for every x < 0",
    "p71_mask_and": "bool values, absent from the open set: comparisons, `&&` of two masks, `where` on a mask; "
                    "`selp` and `shfl` have no .pred form",
    "p91_floordiv_neg": "`x // 2` on i32 with negative x: floor_divide floors (torch, emmy's own folder); a compiler "
                        "copying emmy's CUDA renderer emits C's truncating `/` and is wrong on 15% of the outputs",
    "p92_remainder_neg": "`x % 3` on i32 with negative x: the remainder takes the divisor's sign, not C's",
    "p93_cast_floordiv": "`(x * 4).to(int32) // 3`: a float-to-int truncation, then a floored division; 34% of the "
                         "outputs differ between the two roundings",
}
