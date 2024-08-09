

@group(0) @binding(0) var<storage, read> a : array<a_ty>;
@group(0) @binding(1) var<storage, read> a_strides : array<i64>;

@group(0) @binding(4) var<storage, read_write> c : array<c_ty>;
@group(0) @binding(5) var<storage, read> c_strides : array<i64>;
@group(0) @binding(6) var<storage, read> c_shape : array<i64>;

@compute
@workgroup_size(GRP_SIZE_X, GRP_SIZE_Y, 1)
fn main(
@builtin(workgroup_id) workgroup_id : vec3 <u32>,
@builtin(local_invocation_id) local_id : vec3 <u32>
)
{
    let global_id_x : i64 = i64(workgroup_id.x) * GRP_SIZE_X + i64(local_id.x);

    let tmp : i64 = i64(outer_loop_size) % (NUM_GRP_X * GRP_SIZE_X);
    let start_idx : i64 = global_id_x * (i64(outer_loop_size) / (NUM_GRP_X * GRP_SIZE_X)) + min(global_id_x, tmp);
    var end_idx : i64 = start_idx + (i64(outer_loop_size) / (NUM_GRP_X * GRP_SIZE_X)) + i64(global_id_x < tmp);

}
