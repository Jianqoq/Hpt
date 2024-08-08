

@group(0) @binding(0) var<storage, read> a : array<a_ty>;
@group(0) @binding(1) var<storage, read_write> c : array<c_ty>;

@compute
@workgroup_size(GRP_SIZE_X, GRP_SIZE_Y, 1)
fn main(
@builtin(workgroup_id) workgroup_id : vec3 <u32>,
@builtin(local_invocation_id) local_id : vec3 <u32>
)
{
    let local_row_offset: u32 = local_id.x * GRP_SIZE_Y * NUM_GRP_Y + workgroup_id.x * GRP_SIZE_Y * NUM_GRP_Y;
    let local_col_offset: u32 = workgroup_id.y * GRP_SIZE_Y + local_id.y;
    let t_id : u32 = local_row_offset + local_col_offset;

    let num_threads : u32 = u32(NUM_GRP_X) * u32(GRP_SIZE_X) * u32(NUM_GRP_Y) * u32(GRP_SIZE_Y);
    let tmp : u32 = TOTAL_SIZE % num_threads;
    let start_idx : u32 = t_id * (TOTAL_SIZE / num_threads) + min(t_id, tmp);
    let end_idx : u32 = start_idx + (TOTAL_SIZE / num_threads) + u32(t_id < tmp);

    if end_idx - start_idx == 0 {
        return;
    }

    let range : u32 = end_idx - start_idx;
    for (var i: u32 = 0; i < range; i++) {
        c[t_id + i] = c_ty(t_id + i);
    }
}
