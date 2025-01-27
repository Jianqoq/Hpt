### Type Promote

This document describes the type promotion system in Hpt, which handles type conversions and operations between different numeric types.

### Overview

The type promotion system provides auto casting for different types

### Core Traits

`pub trait FloatOutBinaryPromote<RHS = Self>` See implementation at [here](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/tensor-types/src/type_promote.rs#L59)

`pub trait NormalOutPromote<RHS = Self>` See implementation at [here](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/tensor-types/src/type_promote.rs#L126)

`pub trait SimdCmpPromote<RHS = Self>` See implementation at [here](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/tensor-types/src/type_promote.rs#L312)

`pub trait FloatOutUnaryPromote` See implementation at [here](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/tensor-types/src/type_promote.rs#L575)

### Implementation

The implementation is streightforward. See [here](https://github.com/Jianqoq/Hpt/tree/main/tensor-types/src/promotion)

### Note

Current implementation only supports one type of promotion. Like u32 + i32 = i32. We may want to have multiple version of promotion. The user may want u32 + i32 = i64.
Adding new version is streightforward and just simply using the existing code and do little modification. Adding feature for the promotion, the user enable the specific feature. Then they can use the specific type of promotion.