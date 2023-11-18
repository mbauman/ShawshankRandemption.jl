# This file is a part of Julia. License is MIT: https://julialang.org/license

## Floating number bit twiddling for Uniform sampling in the interval [0, 1)
# 
# The RNG should perform as though a random number were drawn in a theoreticaly
# ideal (0,1) distribution and then rounded down to the previous floating point
# number. Doing this efficiently requires a number of rounding-down bit-twiddling
# utilities.

# We'll be using these a lot...
const sigbits = Base.significand_bits
const sigmask = Base.significand_mask

""" 
    value ÷₂ꜛ N

Divide a positive floating point value by 2^N (N > 0), rounding down.
"""
÷₂ꜛ(value::F, N) where {F <: Base.IEEEFloat} = reinterpret(F, reinterpret(Base.uinttype(F), value) ÷₂ꜛ N)
function ÷₂ꜛ(value::U, N) where U <: Unsigned
    iszero(value) && return value
    F = Base.floattype(U)
    value_exponent = value >> sigbits(F)
    if N < value_exponent
        # We can safely subtract N from the exponent without going subnormal
        ret = value - ((N % U) << sigbits(F))
    else
        new_exponent = value_exponent - N
        # Set an exponent of zero and shift right by the amount we go below the representable exponents
        ret = ((value & sigmask(F)) | (one(U) << sigbits(F))) >> -(new_exponent-1)
    end
    return ret
end

"""
    log2eps(value)

Return the integer log2(eps(value)), rounding down.
"""
log2eps(value::F) where {F <: Base.IEEEFloat} = log2eps(reinterpret(Base.uinttype(F), value))
function log2eps(value::U) where {U <: Unsigned}
    F = Base.floattype(U)
    exp = value >> sigbits(F)
    return exp + iszero(exp) - sigbits(F) - Base.exponent_bias(F)
end

# As a micro-optimization, avoid some pessimizations (guarding against UB) when right-shifting
# by ensuring that the shifts never exceed the number of bits in the type:
shift_mask(::UInt8) = UInt8(7)
shift_mask(::UInt16) = UInt16(15)
shift_mask(::UInt32) = UInt32(31)
shift_mask(::UInt64) = UInt64(63)
shift_mask(::UInt128) = UInt64(127)

"""
    coarse_fixed2float(::Type{F<:IEEEFloat}, r::Unsigned)

Compute r/2^nbits(r), rounding down to either:
    - 0 (if the true result is less than `eps(F)`) or
    - to the previous floating point number

Note that this is guaranteed to produce a normal value (either 0 or a value > eps(F)).
When it returns 0 for a nonzero input, call `fine_fixed2float` to compute the actual value.
"""
function coarse_fixed2float(::Type{F}, r::U) where {F <: Base.IEEEFloat, U <: Unsigned}
    UF = Base.uinttype(F)
    exact_bits = (r + zero(UF)) >> (sizeof(U)*8-sigbits(F))
    value = reinterpret(UF, F(exact_bits)) - (((sigbits(F)) % UF) << sigbits(F))
    omitted_bits = (r << (sigbits(F)))
    coarse_exponent = value >> sigbits(F)
    rawbits_exponent = (Base.exponent_bias(F) - sizeof(U)*8) % UF # treat omitted_bits like the significand of a floating point number
    return (value | (omitted_bits >> ((coarse_exponent - rawbits_exponent) & shift_mask(omitted_bits))) % UF) * !iszero(exact_bits)
end

"""
    fine_fixed2float(::Type{F<:IEEEFloat}, r::Unsigned)

Given an `r` such that `r/2^nbits(r) < eps(F)`, compute `r/2^nbits(r)` rounding down

Note that this may generate a subnormal value or even round down to zero in the general case.
"""
function fine_fixed2float(::Type{F}, r::Unsigned) where {F<:Base.IEEEFloat}
    value = zero(Base.uinttype(F))
    r <<= sigbits(F)
    N = sigbits(F)
    while !iszero(r) && iszero(value)
        value = coarse_fixed2float(F, r) ÷₂ꜛ N
        r <<= sigbits(F)
        N += sigbits(F)
    end
    return value
end
# But we can generally optimize this in many cases
fine_fixed2float(::Type{Float16}, r::UInt8) = UInt16(0)
fine_fixed2float(::Type{Float16}, r::UInt16) = coarse_fixed2float(Float16, r << sigbits(Float16)) ÷₂ꜛ sigbits(Float16) # One iteration, but might go subnormal
fine_fixed2float(::Type{Float32}, r::UInt8) = UInt32(0)
fine_fixed2float(::Type{Float32}, r::UInt16) = UInt32(0)
fine_fixed2float(::Type{Float32}, r::UInt32) = (coarse_fixed2float(Float32, r << sigbits(Float32)) - ((sigbits(Float32) % UInt32) << sigbits(Float32))) * !iszero(r) # One iteration, normal
function fine_fixed2float(::Type{Float32}, r::UInt64)
    # Two iterations max, and we stay normal
    F = Float32
    value = zero(UInt32)
    r <<= sigbits(F)
    (!iszero(r) && iszero(value)) || return value
    value = coarse_fixed2float(F, r)
    value = (value - ((sigbits(F) % UInt32) << sigbits(F))) * !iszero(value)
    r <<= sigbits(F)
    (!iszero(r) && iszero(value)) || return value
    value = coarse_fixed2float(F, r)
    value = (value - ((2*sigbits(F) % UInt32) << sigbits(F))) * !iszero(value)
    return value
end
fine_fixed2float(::Type{Float64}, r::UInt8) = UInt64(0)
fine_fixed2float(::Type{Float64}, r::UInt16) = UInt64(0)
fine_fixed2float(::Type{Float64}, r::UInt32) = UInt64(0)
fine_fixed2float(::Type{Float64}, r::UInt64) = (coarse_fixed2float(Float64, r << sigbits(Float64)) - ((sigbits(Float64) % UInt64) << sigbits(Float64))) * !iszero(r) # One iteration, normal

"""
    fixed2float(::Type{F<:Base.IEEEFloat}, r::Unsigned)

Consider the unsigned value `r` as the fixed-point value `r/2^nbits(r)` and convert
a floating point value, rounding down.
"""
function fixed2float(::Type{F}, r::U) where {F <: Base.IEEEFloat, U <: Unsigned}
    value = coarse_fixed2float(F, r)
    if value == 0
        value = fine_fixed2float(F, r)
    end
    return value
end

reset_significands(oldvals::UInt32, newvals::UInt32) = (oldvals & ~sigmask(Float32) | (newvals & sigmask(Float32)))
function refine_bin(value::UInt64, bits::UInt64)
    neg_exp = Base.exponent_bias(Float64) - (value >> sigbits(Float64))
    return value | (bits >> (neg_exp-sigbits(Float64)))
end
refine_bin(value::UInt32, bits::UInt64) = refine_bin(value, (bits >> 32) % UInt32)
function refine_bin(value::UInt32, bits::UInt32)
    neg_exp = Base.exponent_bias(Float32) - (value >> sigbits(Float32))
    return value | (bits >> (neg_exp-sigbits(Float32)))
end

@inline randf(rng, ::Type{Float16}) = reinterpret(Float16, fixed2float(Float16, rand(rng, UInt32)))
@inline function randf(rng, ::Type{F}) where {F <: Union{Float32, Float64}}
    bits = rand(rng, UInt64)
    value = coarse_fixed2float(F, bits)
    if (F <: Float32) || log2eps(value) > -64 # 
        return reinterpret(F, value)
    elseif (F <: Float64) && value > 0
        return reinterpret(F, refine_bin(value, rand(rng, UInt64)))
    else
        value = fine_fixed2float(F, bits)
        if log2eps(value) > -64
            return reinterpret(F, value)
        elseif value > 0
            return reinterpret(F, refine_bin(value, rand(rng, UInt64)))
        else
            # We won a 1-in-18446744073709551616 lottery
            return naive_randf(rng, F, 64)
        end
    end
end

"""
    bigger +₎ smaller

Add a `smaller` floating point value into bigger, rounding down.
Smaller must be less than the smallest nonzero value of bigger
"""
function +₎(bigger::F, smaller::F) where {F <:Base.IEEEFloat}
    U = Base.uinttype(F)
    reinterpret(F, reinterpret(U, bigger) +₎ reinterpret(U, smaller))
end
function +₎(bigger::U, smaller::U) where {U <: Unsigned}
    F = Base.floattype(U)
    b_exp = bigger >> Base.significand_bits(F)
    s_exp = smaller >> Base.significand_bits(F)
    smaller_fraction = (smaller & Base.significand_mask(F)) | (((s_exp > 0) % U) << Base.significand_bits(F))
    return ifelse(iszero(bigger), smaller, bigger + (smaller_fraction >> (b_exp + iszero(b_exp) - s_exp - iszero(s_exp))))
end
 
naive_randf(rng, ::Type{F}) where {F} = naive_randf(rng, F, UInt64)
function naive_randf(rng, ::Type{F}, ::Type{U}) where {F <: Base.IEEEFloat, U <: Unsigned}
    value = fixed2float(F, rand(rng, U))
    nbits = 8*sizeof(U)
    while log2eps(value) <= -nbits
        value = value +₎ (fixed2float(F, rand(rng, U)) ÷₂ꜛ nbits)
        nbits += 8*sizeof(U)
    end
    return reinterpret(F, value)
end
naive_randf(rng, ::Type{F}, nbits) where {F} = naive_randf(rng, F, UInt64, nbits)
function naive_randf(rng, ::Type{F}, ::Type{U}, nbits) where {F <: Base.IEEEFloat, U <: Unsigned}
    value = fixed2float(F, rand(rng, U)) ÷₂ꜛ nbits
    nbits += 8*sizeof(U)
    while log2eps(value) < -nbits
        value = value +₎ (fixed2float(F, rand(rng, U)) ÷₂ꜛ nbits)
        nbits += 8*sizeof(U)
    end
    return reinterpret(F, value)
end

check(::Type{T}, ::Type{U}) where {T, U} = all(fixed2float(T, u) == T(Float64(u)*2.0^-(sizeof(u)*8), RoundDown) for u in typemin(U):typemax(U))
check(::Type{T}, ::Type{U}) where {T, U<:Union{UInt64, UInt128}} = check_subsample(T, U)
check(::Type{T}, ::Type{U}) where {T<:Float64, U<:Union{UInt8, UInt16}} = all(fixed2float(T, u) == T(Float64(u)*2.0^-(sizeof(u)*8), RoundDown) for u in typemin(U):typemax(U))
check(::Type{T}, ::Type{U}) where {T<:Float64, U<:Union{UInt32}} = check_subsample(T, U)
check(::Type{T}, ::Type{U}) where {T<:Float64, U<:Union{UInt64, UInt128}} = check_subsample(T, U)

function check_subsample(::Type{T}, ::Type{U}) where {T, U}
    for msb in 0:sizeof(U)*8
        for ssb in 0:max(0, msb-1)
            for tsb in 0:max(0, ssb-1)
                base = (one(U) << (msb-1)) | (one(U) << (ssb-1)) | (one(U) << (tsb-1))
                for u in base-U(64):base+U(64)
                    if fixed2float(T, u) != T(big(u)*2.0^-(sizeof(u)*8), RoundDown)
                        println((u, fixed2float(T, u), T(big(u)*2.0^-(sizeof(u)*8), RoundDown)))
                        println("(r, F, FU, U) = ($(repr(u)), $T, $(Base.uinttype(T)), $U)")
                        return false
                    end
                end
            end
        end
    end
    return true
end

function checkall()
    for U in (UInt8, UInt16, UInt32, UInt64, UInt128)
        for F in (Float16, Float32, Float64)
            print("fixed2float(::Type{$F}, ::$U)... ")
            if !check(F, U)
                println()
                debug(F, U)
                return false
            end
            println("✅")
        end
    end
    return true
end


debug(::Type{T}, ::Type{U}) where {T, U} = for u in typemin(U):typemax(U)
    if fixed2float(T, u) != T(big(u)*2.0^-(sizeof(u)*8), RoundDown)
        println((u, fixed2float(T, u), T(big(u)*2.0^-(sizeof(u)*8), RoundDown)))
        println("(r, F, FU, U) = ($(repr(u)), $T, $(Base.uinttype(T)), $U)")
        break
    end
end
