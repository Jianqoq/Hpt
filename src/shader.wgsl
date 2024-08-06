@group(0) @binding(0) var<storage, read> a : array<a_ty>;
@group(0) @binding(1) var<storage, read> a_strides : array<i32>;

@group(0) @binding(2) var<storage, read> b : array<b_ty>;
@group(0) @binding(3) var<storage, read> b_strides : array<i32>;

@group(0) @binding(4) var<storage, read_write> c : array<c_ty>;
@group(0) @binding(5) var<storage, read> c_strides : array<i32>;
@group(0) @binding(6) var<storage, read> c_shape : array<i32>;

@group(0) @binding(7) var<storage, read> outer_loop_size : i32;
@group(0) @binding(8) var<storage, read> inner_loop_size : i32;

@compute
@workgroup_size(GRP_SIZE_X, GRP_SIZE_Y, 1)
fn main(
@builtin(workgroup_id) workgroup_id : vec3 <u32>,
@builtin(local_invocation_id) local_id : vec3 <u32>
)
{
   let cp_id: i32 = PIPELINE_ID;
   let global_id_x: i32 = i32(workgroup_id.x) * GRP_SIZE_X + i32(local_id.x);

   let tmp: i32 = outer_loop_size % (NUM_GRP_X * GRP_SIZE_X);
   let start_idx: i32 = global_id_x * (outer_loop_size / (NUM_GRP_X * GRP_SIZE_X)) + min(global_id_x, tmp);
   var end_idx: i32 = start_idx + (outer_loop_size / (NUM_GRP_X * GRP_SIZE_X)) + i32(global_id_x < tmp);

   if end_idx - start_idx == 0 {
      return;
   }
   var amount : i32 = start_idx * inner_loop_size;
   var c_offset : i32 = 0;
   var a_offset : i32 = 0;
   var b_offset : i32 = 0;
   var prg : array<i32, prg_place_holder>;
   for (var i : i32 = res_ndim - 1; i >= 0; i--)
   {
      let tmp : i32 = amount % c_shape[i];
      c_offset += tmp * c_strides[i];
      a_offset += tmp * a_strides[i];
      b_offset += tmp * b_strides[i];
      prg[i] = tmp;
      amount /= c_shape[i];
   }
   let global_id_y: i32 = i32(workgroup_id.y) * GRP_SIZE_Y + i32(local_id.y);

   let tmp2: i32 = inner_loop_size % (NUM_GRP_Y * GRP_SIZE_Y);
   let start_idx2: i32 = global_id_y * (inner_loop_size / NUM_GRP_Y * GRP_SIZE_Y) + min(global_id_y, tmp2);
   var end_idx2: i32 = start_idx2 + (inner_loop_size / NUM_GRP_Y * GRP_SIZE_Y) + i32(global_id_y < tmp2);

   let c_last_stride: i32 = c_strides[res_ndim - 1];
   let a_last_stride: i32 = a_strides[res_ndim - 1];
   let b_last_stride: i32 = b_strides[res_ndim - 1];

   c_offset += c_last_stride * start_idx2;
   a_offset += a_last_stride * start_idx2;
   b_offset += b_last_stride * start_idx2;

   if end_idx2 - start_idx2 == 0 {
      return;
   }

   let inner_loop_size: i32 = end_idx2 - start_idx2;
   let outer_loop_size: i32 = end_idx - start_idx;

   for (var j : i32 = 0; j < outer_loop_size; j++)
   {
      for (var i : i32 = 0; i < inner_loop_size; i++)
      {
         c[c_offset + i * c_last_stride] = a[a_offset + i * a_last_stride] + b[b_offset + i * b_last_stride];
      }
      for (var i : i32 = res_ndim - 2; i >= 0; i--)
      {
         if (prg[i] + 1 < c_shape[i])
         {
            prg[i]++;
            c_offset += c_strides[i];
            a_offset += a_strides[i];
            b_offset += b_strides[i];
            break;
         }
         else
         {
            prg[i] = i32(0);
            c_offset -= c_strides[i] * (c_shape[i] - 1);
            a_offset -= a_strides[i] * (c_shape[i] - 1);
            b_offset -= b_strides[i] * (c_shape[i] - 1);
         }
      }
   }
}