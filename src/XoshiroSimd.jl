# This file is a part of Julia. License is MIT: https://julialang.org/license

module XoshiroSimd
# Getting the xoroshiro RNG to reliably vectorize is somewhat of a hassle without Simd.jl.
import ..ShawshankRandemption: rand!
using ..ShawshankRandemption: TaskLocalRNG, rand, Xoshiro, CloseOpen01, UnsafeView, SamplerType, SamplerTrivial, getstate, setstate!, bits2float
using Base: BitInteger_types
using Base.Libc: memcpy
using Core.Intrinsics: llvmcall

# Vector-width. Influences random stream.
xoshiroWidth() = Val(8)
# Simd threshold. Influences random stream.
simdThreshold(::Type{T}) where T = 64
simdThreshold(::Type{Bool}) = 640

@inline _rotl45(x::UInt64) = (x<<45)|(x>>19)
@inline _shl17(x::UInt64) = x<<17
@inline _rotl23(x::UInt64) = (x<<23)|(x>>41)
@inline _plus(x::UInt64,y::UInt64) = x+y
@inline _xor(x::UInt64,y::UInt64) = xor(x,y)
@inline _and(x::UInt64, y::UInt64) = x & y
@inline _or(x::UInt64, y::UInt64) = x | y
@inline _lshr(x, y::Int32) = _lshr(x, y % Int64)
@inline _lshr(x::UInt64, y::Int64) = llvmcall("""
    %res = lshr i64 %0, %1
    ret i64 %res
    """,
    UInt64,
    Tuple{UInt64, Int64},
    x, y)

@inline _bits2float(x::UInt64, ::Type{Float64}) = bits2float(x, Float64)

@inline function _bits2float(x::UInt64, ::Type{Float32})
    ui = (x>>>32) % UInt32
    li = x % UInt32
    return (UInt64(bits2float(ui, Float32)) << 32) | UInt64(bits2float(li, Float32))
end

# required operations. These could be written more concisely with `ntuple`, but the compiler
# sometimes refuses to properly vectorize.
for N in [4,8,16]
    let code, s, fshl = "llvm.fshl.v$(N)i64",
        VT = :(NTuple{$N, VecElement{UInt64}})

        s = ntuple(_->VecElement(UInt64(45)), N)
        @eval @inline _rotl45(x::$VT) = ccall($fshl, llvmcall, $VT, ($VT, $VT, $VT), x, x, $s)

        s = ntuple(_->VecElement(UInt64(23)), N)
        @eval @inline _rotl23(x::$VT) = ccall($fshl, llvmcall, $VT, ($VT, $VT, $VT), x, x, $s)

        code = """
        %lshiftOp = shufflevector <1 x i64> <i64 17>, <1 x i64> undef, <$N x i32> zeroinitializer
        %res = shl <$N x i64> %0, %lshiftOp
        ret <$N x i64> %res
        """
        @eval @inline _shl17(x::$VT) = llvmcall($code, $VT, Tuple{$VT}, x)

        code = """
        %res = add <$N x i64> %1, %0
        ret <$N x i64> %res
        """
        @eval @inline _plus(x::$VT, y::$VT) = llvmcall($code, $VT, Tuple{$VT, $VT}, x, y)

        code = """
        %res = xor <$N x i64> %1, %0
        ret <$N x i64> %res
        """
        @eval @inline _xor(x::$VT, y::$VT) = llvmcall($code, $VT, Tuple{$VT, $VT}, x, y)

        code = """
        %res = and <$N x i64> %1, %0
        ret <$N x i64> %res
        """
        @eval @inline _and(x::$VT, y::$VT) = llvmcall($code, $VT, Tuple{$VT, $VT}, x, y)

        code = """
        %res = or <$N x i64> %1, %0
        ret <$N x i64> %res
        """
        @eval @inline _or(x::$VT, y::$VT) = llvmcall($code, $VT, Tuple{$VT, $VT}, x, y)

        code = """
        %tmp = insertelement <1 x i64> undef, i64 %1, i32 0
        %shift = shufflevector <1 x i64> %tmp, <1 x i64> %tmp, <$N x i32> zeroinitializer
        %res = lshr <$N x i64> %0, %shift
        ret <$N x i64> %res
        """
        @eval @inline _lshr(x::$VT, y::Int64) = llvmcall($code, $VT, Tuple{$VT, Int64}, x, y)

        code = """
        %i2 = sub <$N x i64> zeroinitializer, %0
        %i3 = and <$N x i64> %0, %i2
        %i4 = uitofp <$N x i64> %i3 to <$N x double>
        %i5 = bitcast <$N x double> %i4 to <$N x i64>
        %i6 = sub <$N x i64> <$(join(fill("i64 9209861237972664320", N), ", "))>, %i5
        %i7 = icmp eq <$N x i64> %0, zeroinitializer
        %i8 = select <$N x i1> %i7, <$N x i64> zeroinitializer, <$N x i64> %i6
        %i9 = xor <$N x i64> %0, %i3
        %i10 = lshr <$N x i64> %i9, <$(join(fill("i64 12", N), ", "))>
        %i11 = or <$N x i64> %i8, %i10
        ret <$N x i64> %i11
        """
        @eval @inline _bits2float(x::$VT, ::Type{Float64}) = llvmcall($code, $VT, Tuple{$VT}, x)

        code = """
        %i1 = bitcast <$N x i64> %0 to <$(2N) x i32>
        %i2 = sub <$(2N) x i32> zeroinitializer, %i1
        %i3 = and <$(2N) x i32> %i1, %i2
        %i4 = uitofp <$(2N) x i32> %i3 to <$(2N) x float>
        %i5 = bitcast <$(2N) x float> %i4 to <$(2N) x i32>
        %i6 = sub <$(2N) x i32> <$(join(fill("i32 2122317824", 2N), ", "))>, %i5
        %i7 = icmp eq <$(2N) x i32> %i1, zeroinitializer
        %i8 = select <$(2N) x i1> %i7, <$(2N) x i32> zeroinitializer, <$(2N) x i32> %i6
        %i9 = xor <$(2N) x i32> %i1, %i3
        %i10 = lshr <$(2N) x i32> %i9, <$(join(fill("i32 9", 2N), ", "))>
        %i11 = or <$(2N) x i32> %i8, %i10
        %i12 = bitcast <$(2N) x i32> %i11 to <$N x i64>
        ret <$N x i64> %i12
        """
        @eval @inline _bits2float(x::$VT, ::Type{Float32}) = llvmcall($code, $VT, Tuple{$VT}, x)

        code = """
        %i1 = bitcast <$N x i64> %0 to <$(2N) x i32>
        %i2 = icmp ult <$(2N) x i32> %i1, <$(join(fill("i32 989855744", 2N), ", "))>
        %i3 = bitcast <$(2N) x i1> %i2 to i$(2N)
        %i4 = icmp ne i$(2N) %i3, 0
        %i5 = zext i1 %i4 to i8
        ret i8 %i5
        """
        @eval @inline _any_below_support(x::$VT) = llvmcall($code, Bool, Tuple{$VT}, x)
    end
end


function forkRand(rng::Union{TaskLocalRNG, Xoshiro}, ::Val{N}) where N
    # constants have nothing up their sleeve. For more discussion, cf rng_split in task.c
    # 0x02011ce34bce797f == hash(UInt(1))|0x01
    # 0x5a94851fb48a6e05 == hash(UInt(2))|0x01
    # 0x3688cf5d48899fa7 == hash(UInt(3))|0x01
    # 0x867b4bb4c42e5661 == hash(UInt(4))|0x01
    s0 = ntuple(i->VecElement(0x02011ce34bce797f * rand(rng, UInt64)), Val(N))
    s1 = ntuple(i->VecElement(0x5a94851fb48a6e05 * rand(rng, UInt64)), Val(N))
    s2 = ntuple(i->VecElement(0x3688cf5d48899fa7 * rand(rng, UInt64)), Val(N))
    s3 = ntuple(i->VecElement(0x867b4bb4c42e5661 * rand(rng, UInt64)), Val(N))
    (s0, s1, s2, s3)
end

_id(x, T) = x

@inline function xoshiro_bulk(rng::Union{TaskLocalRNG, Xoshiro}, dst::Ptr{UInt8}, len::Int, T::Union{Type{UInt8}, Type{Bool}, Type{Float32}, Type{Float64}}, ::Val{N}, f::F = _id) where {N, F}
    if len >= simdThreshold(T)
        written = xoshiro_bulk_simd(rng, dst, len, T, Val(N), f)
        len -= written
        dst += written
    end
    if len != 0
        xoshiro_bulk_nosimd(rng, dst, len, T, f)
    end
    nothing
end

@noinline function xoshiro_bulk_nosimd(rng::Union{TaskLocalRNG, Xoshiro}, dst::Ptr{UInt8}, len::Int, ::Type{T}, f::F
                                       ) where {T, F}
    s0, s1, s2, s3 = getstate(rng)
    i = 0
    while i+8 <= len
        res = _plus(_rotl23(_plus(s0,s3)),s0)
        unsafe_store!(reinterpret(Ptr{UInt64}, dst + i), f(res, T))
        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
        i += 8
    end
    if i < len
        res = _plus(_rotl23(_plus(s0,s3)),s0)
        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
        ref = Ref(f(res, T))
        # TODO: This may make the random-stream dependent on system endianness
        GC.@preserve ref memcpy(dst+i, Base.unsafe_convert(Ptr{Cvoid}, ref), len-i)
    end
    setstate!(rng, (s0, s1, s2, s3, nothing))
    nothing
end

@noinline function xoshiro_bulk_nosimd(rng::Union{TaskLocalRNG, Xoshiro}, dst::Ptr{UInt8}, len::Int, ::Type{Bool}, f)
    s0, s1, s2, s3 = getstate(rng)
    i = 0
    while i+8 <= len
        res = _plus(_rotl23(_plus(s0,s3)),s0)
        shift = 0
        while i+8 <= len && shift < 8
            resLoc = _and(_lshr(res, shift), 0x0101010101010101)
            unsafe_store!(reinterpret(Ptr{UInt64}, dst + i), resLoc)
            i += 8
            shift += 1
        end

        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
    end
    if i < len
        # we may overgenerate some bytes here, if len mod 64 <= 56 and len mod 8 != 0
        res = _plus(_rotl23(_plus(s0,s3)),s0)
        resLoc = _and(res, 0x0101010101010101)
        ref = Ref(resLoc)
        GC.@preserve ref memcpy(dst+i, Base.unsafe_convert(Ptr{Cvoid}, ref), len-i)
        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
    end
    setstate!(rng, (s0, s1, s2, s3, nothing))
    nothing
end

@noinline function outlined_rare_branch(s0, s1, s2, s3)
    res = _plus(_rotl23(_plus(s0,s3)),s0)
    t = _shl17(s1)
    s2 = _xor(s2, s0)
    s3 = _xor(s3, s1)
    s1 = _xor(s1, s2)
    s0 = _xor(s0, s3)
    s2 = _xor(s2, t)
    s3 = _rotl45(s3)
    new1 = map(x->bits2float(x.value, Float32), res)
    res = _plus(_rotl23(_plus(s0,s3)),s0)
    t = _shl17(s1)
    s2 = _xor(s2, s0)
    s3 = _xor(s3, s1)
    s1 = _xor(s1, s2)
    s0 = _xor(s0, s3)
    s2 = _xor(s2, t)
    s3 = _rotl45(s3)
    new2 = map(x->bits2float(x.value, Float32), res)
    vals = reinterpret(NTuple{length(res), UInt64}, (new1..., new2...))
    return s0, s1, s2, s3, vals
end

@noinline function xoshiro_bulk_simd(rng::Union{TaskLocalRNG, Xoshiro}, dst::Ptr{UInt8}, len::Int, ::Type{Float32}, ::Val{N}, f::F) where {N,F}
    T = Float32
    s0, s1, s2, s3 = forkRand(rng, Val(N))

    i = 0
    while i + 8*N <= len
        res = _plus(_rotl23(_plus(s0,s3)),s0)
        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
        vals = f(res, T)
        if _any_below_support(vals)
            # At least one value is missing random bits from
            # at least its mantissa; we need more bits!
            res = _plus(_rotl23(_plus(s0,s3)),s0)
            t = _shl17(s1)
            s2 = _xor(s2, s0)
            s3 = _xor(s3, s1)
            s1 = _xor(s1, s2)
            s0 = _xor(s0, s3)
            s2 = _xor(s2, t)
            s3 = _rotl45(s3)
            new_mantissas =  _reset_mantissas(vals, res)
            if _any_are_zero(vals)
                # we have to add dynamic range below 2^-32 to at least one value
                new_values = _div2carat32(f(res, T)) # -32<<23
                vals = _ifelse(iszero, new_values, new_mantissas)
            else
                # this is the very common case: the nonzero exponents are all valid;
                # just refresh all mantissa bits in a SIMD-friendly way
                vals = new_mantissas
            end
        end
        unsafe_store!(reinterpret(Ptr{NTuple{N,VecElement{UInt64}}}, dst + i), vals)
        i += 8*N
    end
    return i
end

@noinline function xoshiro_bulk_simd(rng::Union{TaskLocalRNG, Xoshiro}, dst::Ptr{UInt8}, len::Int, ::Type{T}, ::Val{N}, f::F) where {T,N,F}
    s0, s1, s2, s3 = forkRand(rng, Val(N))

    i = 0
    while i + 8*N <= len
        res = _plus(_rotl23(_plus(s0,s3)),s0)
        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
        unsafe_store!(reinterpret(Ptr{NTuple{N,VecElement{UInt64}}}, dst + i), f(res, T))
        i += 8*N
    end
    return i
end

@noinline function xoshiro_bulk_simd(rng::Union{TaskLocalRNG, Xoshiro}, dst::Ptr{UInt8}, len::Int, ::Type{Bool}, ::Val{N}, f) where {N}
    s0, s1, s2, s3 = forkRand(rng, Val(N))
    msk = ntuple(i->VecElement(0x0101010101010101), Val(N))
    i = 0
    while i + 64*N <= len
        res = _plus(_rotl23(_plus(s0,s3)),s0)
        t = _shl17(s1)
        s2 = _xor(s2, s0)
        s3 = _xor(s3, s1)
        s1 = _xor(s1, s2)
        s0 = _xor(s0, s3)
        s2 = _xor(s2, t)
        s3 = _rotl45(s3)
        for k=0:7
            tmp = _lshr(res, k)
            toWrite = _and(tmp, msk)
            unsafe_store!(reinterpret(Ptr{NTuple{N,VecElement{UInt64}}}, dst + i + k*N*8), toWrite)
        end
        i += 64*N
    end
    return i
end


function rand!(rng::Union{TaskLocalRNG, Xoshiro}, dst::Array{Float32}, ::SamplerTrivial{CloseOpen01{Float32}})
    GC.@preserve dst xoshiro_bulk(rng, convert(Ptr{UInt8}, pointer(dst)), length(dst)*4, Float32, xoshiroWidth(), _bits2float)
    dst
end

function rand!(rng::Union{TaskLocalRNG, Xoshiro}, dst::Array{Float64}, ::SamplerTrivial{CloseOpen01{Float64}})
    GC.@preserve dst xoshiro_bulk(rng, convert(Ptr{UInt8}, pointer(dst)), length(dst)*8, Float64, xoshiroWidth(), _bits2float)
    dst
end

for T in BitInteger_types
    @eval function rand!(rng::Union{TaskLocalRNG, Xoshiro}, dst::Union{Array{$T}, UnsafeView{$T}}, ::SamplerType{$T})
        GC.@preserve dst xoshiro_bulk(rng, convert(Ptr{UInt8}, pointer(dst)), length(dst)*sizeof($T), UInt8, xoshiroWidth())
        dst
    end
end

function rand!(rng::Union{TaskLocalRNG, Xoshiro}, dst::Array{Bool}, ::SamplerType{Bool})
    GC.@preserve dst xoshiro_bulk(rng, convert(Ptr{UInt8}, pointer(dst)), length(dst), Bool, xoshiroWidth())
    dst
end

end # module
