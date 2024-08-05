@group(0) @binding(0) var<storage, read> a : array<u32>;
@group(0) @binding(1) var<storage, read> b : array<u32>;
@group(0) @binding(2) var<storage, read> a_strides : array<i64>;
@group(0) @binding(3) var<storage, read> b_strides : array<i64>;

@group(0) @binding(4) var<storage, read_write> c : array<u32>;
@group(0) @binding(5) var<storage, read> c_strides : array<i64>;

@compute
@workgroup_size(256)
fn main(
      @builtin(global_invocation_id) global_id : vec3 < u32>,
      @builtin(workgroup_id) workgroup_id : vec3 < u32>,
      @builtin(local_invocation_id) local_id : vec3 < u32>
)
{
   c[global_id.y] = workgroup_id.x;
}
