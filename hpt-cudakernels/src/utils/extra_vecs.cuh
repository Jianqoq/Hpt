#pragma once
#include "cuda_fp16.h"
#include "cuda_bf16.h"

struct half3
{
    __half x;
    __half y;
    __half z;
};

struct __align__(8) half4
{
    __half x;
    __half y;
    __half z;
    __half w;
};

struct bf163
{
    __nv_bfloat16 x;
    __nv_bfloat16 y;
    __nv_bfloat16 z;
};

struct __align__(8) bf164
{
    __nv_bfloat16 x;
    __nv_bfloat16 y;
    __nv_bfloat16 z;
    __nv_bfloat16 w;
};

struct bool2
{
    bool x;
    bool y;
};

struct bool3
{
    bool x;
    bool y;
    bool z;
};

struct bool4
{
    bool x;
    bool y;
    bool z;
    bool w;
};
