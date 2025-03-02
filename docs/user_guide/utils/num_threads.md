# set_num_threads

```rust
set_num_threads(num_threads: usize)
```

Set the parallelism thread numbers
## Parameters:
`num_threads`: number of threads to use

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |

# get_num_threads

```rust
get_num_threads()
```

Get the current number of threads using

## Backend Support
| Backend | Supported |
|---------|-----------|
| CPU     | ✅         |
| Cuda    | ❌        |