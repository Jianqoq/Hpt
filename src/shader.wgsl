@group(0) @binding(0) var<storage, read> a : array<i64>;
@group(0) @binding(1) var<storage, read> a_strides : array<i64>;
@group(0) @binding(2) var<storage, read> a_shape : array<i64>;

@group(0) @binding(3) var<storage, read> b : array<i64>;
@group(0) @binding(4) var<storage, read> b_strides : array<i64>;
@group(0) @binding(5) var<storage, read> b_shape : array<i64>;

@group(0) @binding(6) var<storage, read_write> c : array<i64>;
@group(0) @binding(7) var<storage, read> c_strides : array<i64>;
@group(0) @binding(8) var<storage, read> c_shape : array<i64>;

@group(0) @binding(9) var<storage, read> outer_loop_size : i64;
@group(0) @binding(10) var<storage, read> inner_loop_size : i64;

@group(0) @binding(11) var<storage, read> res_ndim : i64;

@compute
@workgroup_size(16, 16, 1)
fn main(
@builtin(workgroup_id) workgroup_id : vec3 <u32>,
@builtin(local_invocation_id) local_id : vec3 <u32>
)
{
   let global_id_x = i64(workgroup_id.x) * 16 + i64(local_id.x);

   let tmp = outer_loop_size % (1024 * 16);
   let start_idx = global_id_x * (outer_loop_size / (1024 * 16)) + min(global_id_x, tmp);
   var end_idx = start_idx + (outer_loop_size / (1024 * 16)) + i64(global_id_x < tmp);

   if end_idx - start_idx == 0 {
      return;
   }
   var amount : i64 = start_idx * inner_loop_size;
   var c_offset : i64 = 0;
   var a_offset : i64 = 0;
   var b_offset : i64 = 0;
   for (var i : i64 = res_ndim - 1; i >= 0; i--)
   {
      let tmp : i64 = amount % c_shape[i];
      c_offset += tmp * c_strides[i];
      a_offset += tmp * a_strides[i];
      b_offset += tmp * b_strides[i];
      amount /= c_shape[i];
   }

   let workgroup_id_y = i64(workgroup_id.y);
   let local_id_y = i64(local_id.y);
   let global_id_y = workgroup_id_y * 16 + local_id_y;

   let tmp2 = inner_loop_size % (1024 * 16);
   let start_idx2 = global_id_y * (inner_loop_size / 1024 * 16) + min(global_id_y, tmp2);
   var end_idx2 = start_idx2 + (inner_loop_size / 1024 * 16) + i64(global_id_y < tmp2);

   c[global_id_x] = i64(workgroup_id.y);
   var c_offset2 : i64 = c_strides[res_ndim - 1] * start_idx2;
   var a_offset2 : i64 = a_strides[res_ndim - 1] * start_idx2;
   var b_offset2 : i64 = b_strides[res_ndim - 1] * start_idx2;
}
