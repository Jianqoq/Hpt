__device__ bool is_contiguous(
    const size_t num_dims,
    const size_t *strides)
{
    size_t prev_stride = strides[num_dims - 1];
    for (size_t i = num_dims - 2; i >= 0; i--)
    {
        if (strides[i] < prev_stride)
        {
            return false;
        }
        prev_stride = strides[i];
    }
    return true;
}

__device__ bool shape_equal(size_t ndim, size_t *shape, size_t *lhs_shape, size_t *rhs_shape)
{
    for (size_t i = 0; i < ndim; i++)
    {
        if (shape[i] != lhs_shape[i] || shape[i] != rhs_shape[i])
        {
            return false;
        }
    }
    return true;
}