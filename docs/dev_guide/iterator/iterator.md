### Iterator

Iterator are implemented using Rayon trait `UnindexedProducer`, the tasks are splitted in [split](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-iterator/src/par_strided.rs#L541) method. The main loop is happened in [fold_with](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-iterator/src/par_strided_zip.rs#L471). Iterator can be used to implement elementwise or broadcast elementwise calculations. Usage can be found at [here](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-core/src/ops/cpu/utils/binary/binary_normal.rs#L46) and [here](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-core/src/ops/cpu/utils/unary/unary.rs#L19)

### Known issue

1. Current iterator like `ParStrided`, the `fold_with` method doesn't have any looping logic. Maybe we can write same logic as `ParStridedZip` in `fold_with`.

2. When outer loop size is 1, there will be no parallelism because the tasks splits based on the outer loop size.