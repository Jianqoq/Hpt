# Scalar Types

This document describes the scalar type system in Hpt.

## Supported Types

Hpt implements math methods for the following types:

```rust
bool
i8
u8
i16
u16
i32
u32
i64
u64
f32
f64
half::f16
half::bf16
Complex32
Complex64
```

### Common Traits
`pub trait FloatOutBinary<RHS = Self>` See implementation at [FloatOutBinary](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L49)

`pub trait FloatOutBinary2` See implementation at [FloatOutBinary2](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L65)

`pub trait NormalOut<RHS = Self>` See implementation at [NormalOut](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L78)

`pub trait NormalOut2` See implementation at [NormalOut2](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L103)

`pub trait NormalOutUnary` See implementation at [NormalOutUnary](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L141)

`pub trait NormalOutUnary2` See implementation at [NormalOutUnary2](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L176)

`pub trait BitWiseOut<RHS = Self>` See implementation at [BitWiseOut](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L213)

`pub trait BitWiseOut2` See implementation at [BitWiseOut2](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L231)

`pub trait Cmp<RHS = Self>` See implementation at [Cmp](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L251)

`pub trait SimdCmp<RHS = Self>` See implementation at [SimdCmp](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L270)

`pub trait Eval` See implementation at [Eval](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L320)

`pub trait Eval2` See implementation at [Eval2](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L332)

`pub trait FloatOutUnary` See implementation at [FloatOutUnary](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L348)

`pub trait FloatOutUnary2` See implementation at [FloatOutUnary2](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L463)

`pub trait Cast<T>` See implementation at [Cast](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/cast.rs#L6)

### Usage

In hpt-core, all the computation will be done by using these methods. If you are a developer want to implement a new method for Tensor, you will need to use these traits bounds. If you found that implement your new method for Tensor requires to write tons of trait bounds, you probabaly want to implement the method for scalar directly, then use this scalar method in Tensor impl by simply added one trait bound.

### Simd

All the scalar implemented also has their own simd type. It can be 128bit, 256bit, or 512bit, it depends on the machine.

### Known issue

Since I don't have avx512 CPU in hand, I can't test and make sure the 512bit code can compile.