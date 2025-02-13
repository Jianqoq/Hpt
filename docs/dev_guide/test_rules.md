### Test cases

All the operators implemented must be tested in 4 cases.
1. contiguous Tensor [see](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-tests/src/hpt/cpu/reduce.rs#L143)
2. uncontiguous Tensor [see](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-tests/src/hpt/cpu/reduce.rs#L198)
3. sliced contiguous Tensor [see](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-tests/src/hpt/cpu/reduce.rs#L241)
4. sliced uncontiguous Tensor [see](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-tests/src/hpt/cpu/reduce.rs#L333)

Besides that, all the algorithm must be tested by using random input and random arguments.