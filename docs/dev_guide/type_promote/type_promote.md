### Type Promote

The type promotion system in Hpt handles type conversions and operations between different numeric types.

### Overview

The type promotion system provides auto casting for different types

### Core Traits

`pub trait FloatOutBinaryPromote<RHS = Self>` See implementation at [here](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L59)

`pub trait NormalOutPromote<RHS = Self>` See implementation at [here](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L126)

`pub trait SimdCmpPromote<RHS = Self>` See implementation at [here](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L312)

`pub trait FloatOutUnaryPromote` See implementation at [here](https://github.com/Jianqoq/Hpt/blob/d9a51874b3447d562b7c9d043b50eb05259b78c4/hpt-types/src/type_promote.rs#L575)

### Implementation

The implementation is straightforward. See [here](https://github.com/Jianqoq/Hpt/tree/main/hpt-types/src/promotion)

### Note

Current implementation only supports one type of promotion. Like u32 + i32 = i32. We may want to have multiple version of promotion. The user may want u32 + i32 = i64.
Adding new version is straightforward and just simply using the existing code and do little modification. 

### How
1. create a new feature that the user can choose in file `hpt-types/Cargo.toml`
```cargo
[features]
.. // features
default = [.., "normal_promote"]
normal_promote = []
// new_promote = []
```

2. Go to `hpt-types/src/promotion`, create a new folder, like `new_promote/`

3. create files under `hpt-types/src/promotion/new_promote/` like how we create in `hpt-types/src/promotion/normal_promote/`, then write promotion logic for the new promote.

4. Go to `lib.rs` and add new `mod` in promotion mod
```rust
pub mod promotion {
    #[cfg(feature = "normal_promote")]
    pub(crate) mod normal_promote {
        ..
    }
    #[cfg(feature = "new_promote")]
    pub(crate) mod new_promote {
        ..
    }
    pub(crate) mod utils;
}
```

Adding feature for the promotion, the user enable the specific feature. Then they can use the specific type of promotion.