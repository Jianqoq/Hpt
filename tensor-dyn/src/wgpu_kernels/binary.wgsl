

@group(0) @binding(0) var<storage, read> a : array<a_ty>;
@group(0) @binding(1) var<storage, read> a_strides : array<i64>;

@group(0) @binding(2) var<storage, read> b : array<b_ty>;
@group(0) @binding(3) var<storage, read> b_strides : array<i64>;

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

    if end_idx - start_idx == 0 {
        return;
    }
    var amount : i64 = start_idx * i64(inner_loop_size);
    var c_offset : i64 = 0;
    var a_offset : i64 = 0;
    var b_offset : i64 = 0;
    var prg : array<i64, res_ndim>;
    for (var i : i64 = res_ndim - 1; i >= 0; i--)
    {
        let idx : u32 = u32(i);
        let tmp : i64 = amount % c_shape[idx];
        c_offset += tmp * c_strides[idx];
        a_offset += tmp * a_strides[idx];
        b_offset += tmp * b_strides[idx];
        prg[idx] = tmp;
        amount /= c_shape[idx];
    }
    let global_id_y : i64 = i64(workgroup_id.y) * GRP_SIZE_Y + i64(local_id.y);

    let tmp2 : i64 = i64(inner_loop_size) % (NUM_GRP_Y * GRP_SIZE_Y);
    let start_idx2 : i64 = global_id_y * (i64(inner_loop_size) / NUM_GRP_Y * GRP_SIZE_Y) + min(global_id_y, tmp2);
    var end_idx2 : i64 = start_idx2 + (i64(inner_loop_size) / NUM_GRP_Y * GRP_SIZE_Y) + i64(global_id_y < tmp2);

    if end_idx2 - start_idx2 == 0 {
        return;
    }

    c_offset += c_last_stride * start_idx2;
    a_offset += a_last_stride * start_idx2;
    b_offset += b_last_stride * start_idx2;

    let inner : i64 = end_idx2 - start_idx2;
    let outer : i64 = end_idx - start_idx;

    for (var j : i64 = 0; j < outer; j++)
    {
        for (var i : i64 = 0; i < inner; i++)
        {
            c[c_offset + i * c_last_stride] = c_ty(a[a_offset + i * a_last_stride]) op_place_holder c_ty(b[b_offset + i * b_last_stride]);
        }
        for (var k : i64 = res_ndim - 2; k >= 0; k--)
        {
            let idx : u32 = u32(k);
            if (prg[idx] + 1 < c_shape[idx])
            {
                prg[idx]++;
                c_offset += c_strides[idx];
                a_offset += a_strides[idx];
                b_offset += b_strides[idx];
                break;
            }
            else
            {
                prg[idx] = i64(0);
                c_offset -= c_strides[idx] * (c_shape[idx] - 1);
                a_offset -= a_strides[idx] * (c_shape[idx] - 1);
                b_offset -= b_strides[idx] * (c_shape[idx] - 1);
            }
        }
    }
}
