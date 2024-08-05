@group(0) @binding(0) var<storage, read> a : array<u32>;
@group(0) @binding(1) var<storage, read> a_strides : array<i64>;
@group(0) @binding(2) var<storage, read> a_shape : array<i64>;

@group(0) @binding(3) var<storage, read> b : array<u32>;
@group(0) @binding(4) var<storage, read> b_strides : array<i64>;
@group(0) @binding(5) var<storage, read> b_shape : array<i64>;

@group(0) @binding(6) var<storage, read_write> c : array<u32>;
@group(0) @binding(7) var<storage, read> c_strides : array<i64>;
@group(0) @binding(8) var<storage, read> c_shape : array<i64>;

@compute
@workgroup_size(16, 16, 1)
fn main(
      @builtin(workgroup_id) workgroup_id : vec3 < u32>,
      @builtin(local_invocation_id) local_id : vec3 < u32>
)
{
   let workgroup_id_x = u64(workgroup_id.x);
   let local_id_x = u64(local_id.x);
   let global_id = workgroup_id_x * 16 + local_id_x;
   if (global_id >= 1024) {
      return;
   }
   c[workgroup_id.x] = workgroup_id.x;
}
