### Pointer

Hpt is using multi threading across the whole operators implementation. However, raw pointer can't be send to threads safely.

So we created a [wrapper](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-common/src/utils/pointer.rs#L11) for pointer.

In the whole project, almost all the parallel iteration are using [wrapper](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-common/src/utils/pointer.rs#L11) to read and write data.

You may notice there is a `bound_check` feature, however, this feature is not fully tested and may not reliable. This feature need to stablize.