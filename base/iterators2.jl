pairs(::IndexCartesian, A::AbstractArray) = Pairs(A, Base.CartesianIndices(axes(A)))
pairs(A::AbstractArray)  = pairs(IndexCartesian(), A)

@doc """
    partition(collection, n)

Iterate over a collection `n` elements at a time.

# Examples
```jldoctest
julia> collect(Iterators.partition([1,2,3,4,5], 2))
3-element Vector{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}}:
 [1, 2]
 [3, 4]
 [5]
```
""" function partition(c, n::Integer)
    n < 1 && throw(ArgumentError("cannot create partitions of length $n"))
    return PartitionIterator(c, Int(n))
end

struct PartitionIterator{T}
    c::T
    n::Int
end
# Partitions are explicitly a linear indexing operation, so reshape to 1-d immediately
PartitionIterator(A::AbstractArray, n::Int) = PartitionIterator(Base.vec(A), n)
PartitionIterator(v::AbstractVector, n::Int) = PartitionIterator{typeof(v)}(v, n)

eltype(::Type{PartitionIterator{T}}) where {T} = Vector{eltype(T)}
# Arrays use a generic `view`-of-a-`vec`, so we cannot exactly predict what we'll get back
eltype(::Type{PartitionIterator{T}}) where {T<:AbstractArray} = AbstractVector{eltype(T)}
# But for some common implementations in Base we know the answer exactly
eltype(::Type{PartitionIterator{T}}) where {T<:Vector} = SubArray{eltype(T), 1, T, Tuple{UnitRange{Int}}, true}

IteratorEltype(::Type{PartitionIterator{T}}) where {T} = IteratorEltype(T)
IteratorEltype(::Type{PartitionIterator{T}}) where {T<:AbstractArray} = EltypeUnknown()
IteratorEltype(::Type{PartitionIterator{T}}) where {T<:Vector} = IteratorEltype(T)

partition_iteratorsize(::HasShape) = HasLength()
partition_iteratorsize(isz) = isz
function IteratorSize(::Type{PartitionIterator{T}}) where {T}
    partition_iteratorsize(IteratorSize(T))
end

function length(itr::PartitionIterator)
    l = length(itr.c)
    return cld(l, itr.n)
end

function iterate(itr::PartitionIterator{<:AbstractRange}, state = firstindex(itr.c))
    state > lastindex(itr.c) && return nothing
    r = min(state + itr.n - 1, lastindex(itr.c))
    return @inbounds itr.c[state:r], r + 1
end

function iterate(itr::PartitionIterator{<:AbstractArray}, state = firstindex(itr.c))
    state > lastindex(itr.c) && return nothing
    r = min(state + itr.n - 1, lastindex(itr.c))
    return @inbounds view(itr.c, state:r), r + 1
end

struct IterationCutShort; end

function iterate(itr::PartitionIterator, state...)
    # This is necessary to remember whether we cut the
    # last element short. In such cases, we do return that
    # element, but not the next one
    state === (IterationCutShort(),) && return nothing
    v = Vector{eltype(itr.c)}(undef, itr.n)
    i = 0
    y = iterate(itr.c, state...)
    while y !== nothing
        i += 1
        v[i] = y[1]
        if i >= itr.n
            break
        end
        y = iterate(itr.c, y[2])
    end
    i === 0 && return nothing
    return resize!(v, i), y === nothing ? IterationCutShort() : y[2]
end

@doc """
    Stateful(itr)

There are several different ways to think about this iterator wrapper:

1. It provides a mutable wrapper around an iterator and
   its iteration state.
2. It turns an iterator-like abstraction into a `Channel`-like
   abstraction.
3. It's an iterator that mutates to become its own rest iterator
   whenever an item is produced.

`Stateful` provides the regular iterator interface. Like other mutable iterators
(e.g. [`Base.Channel`](@ref)), if iteration is stopped early (e.g. by a [`break`](@ref) in a [`for`](@ref) loop),
iteration can be resumed from the same spot by continuing to iterate over the
same iterator object (in contrast, an immutable iterator would restart from the
beginning).

# Examples
```jldoctest
julia> a = Iterators.Stateful("abcdef");

julia> isempty(a)
false

julia> popfirst!(a)
'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)

julia> collect(Iterators.take(a, 3))
3-element Vector{Char}:
 'b': ASCII/Unicode U+0062 (category Ll: Letter, lowercase)
 'c': ASCII/Unicode U+0063 (category Ll: Letter, lowercase)
 'd': ASCII/Unicode U+0064 (category Ll: Letter, lowercase)

julia> collect(a)
2-element Vector{Char}:
 'e': ASCII/Unicode U+0065 (category Ll: Letter, lowercase)
 'f': ASCII/Unicode U+0066 (category Ll: Letter, lowercase)

julia> Iterators.reset!(a); popfirst!(a)
'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)

julia> Iterators.reset!(a, "hello"); popfirst!(a)
'h': ASCII/Unicode U+0068 (category Ll: Letter, lowercase)
```

```jldoctest
julia> a = Iterators.Stateful([1,1,1,2,3,4]);

julia> for x in a; x == 1 || break; end

julia> peek(a)
3

julia> sum(a) # Sum the remaining elements
7
```
"""
mutable struct Stateful{T, VS}
    itr::T
    # A bit awkward right now, but adapted to the new iteration protocol
    nextvalstate::Union{VS, Nothing}
    @inline function Stateful{<:Any, Any}(itr::T) where {T}
        return new{T, Any}(itr, iterate(itr))
    end
    @inline function Stateful(itr::T) where {T}
        VS = approx_iter_type(T)
        return new{T, VS}(itr, iterate(itr)::VS)
    end
end

function reset!(s::Stateful)
    setfield!(s, :nextvalstate, iterate(s.itr)) # bypass convert call of setproperty!
    return s
end
function reset!(s::Stateful{T}, itr::T) where {T}
    s.itr = itr
    reset!(s)
    return s
end


# Try to find an appropriate type for the (value, state tuple),
# by doing a recursive unrolling of the iteration protocol up to
# fixpoint.
approx_iter_type(itrT::Type) = _approx_iter_type(itrT, Base._return_type(iterate, Tuple{itrT}))
# Not actually called, just passed to return type to avoid
# having to typesplit on Nothing
function doiterate(itr, valstate::Union{Nothing, Tuple{Any, Any}})
    valstate === nothing && return nothing
    val, st = valstate
    return iterate(itr, st)
end
function _approx_iter_type(itrT::Type, vstate::Type)
    vstate <: Union{Nothing, Tuple{Any, Any}} || return Any
    vstate <: Union{} && return Union{}
    itrT <: Union{} && return Union{}
    nextvstate = Base._return_type(doiterate, Tuple{itrT, vstate})
    return (nextvstate <: vstate ? vstate : Any)
end

Stateful(x::Stateful) = x
convert(::Type{Stateful}, itr) = Stateful(itr)
@inline isdone(s::Stateful, st=nothing) = s.nextvalstate === nothing

@inline function popfirst!(s::Stateful)
    vs = s.nextvalstate
    if vs === nothing
        throw(Base.EOFError())
    else
        val, state = vs
        Core.setfield!(s, :nextvalstate, iterate(s.itr, state))
        return val
    end
end

@inline function peek(s::Stateful, sentinel=nothing)
    ns = s.nextvalstate
    return ns !== nothing ? ns[1] : sentinel
end
@inline iterate(s::Stateful, state=nothing) = s.nextvalstate === nothing ? nothing : (popfirst!(s), nothing)
IteratorSize(::Type{<:Stateful{T}}) where {T} = IteratorSize(T) isa IsInfinite ? IsInfinite() : SizeUnknown()
eltype(::Type{<:Stateful{T}}) where {T} = eltype(T)
IteratorEltype(::Type{<:Stateful{T}}) where {T} = IteratorEltype(T)

"""
    only(x)

Return the one and only element of collection `x`, or throw an [`ArgumentError`](@ref) if the
collection has zero or multiple elements.

See also [`first`](@ref), [`last`](@ref).

!!! compat "Julia 1.4"
    This method requires at least Julia 1.4.

# Examples
```jldoctest
julia> only(["a"])
"a"

julia> only("a")
'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)

julia> only(())
ERROR: ArgumentError: Tuple contains 0 elements, must contain exactly 1 element
Stacktrace:
[...]

julia> only(('a', 'b'))
ERROR: ArgumentError: Tuple contains 2 elements, must contain exactly 1 element
Stacktrace:
[...]
```
"""
@propagate_inbounds only(x) = _only(x, iterate)

@propagate_inbounds function _only(x, ::typeof(iterate))
    i = iterate(x)
    @boundscheck if i === nothing
        throw(ArgumentError("Collection is empty, must contain exactly 1 element"))
    end
    (ret, state) = i::NTuple{2,Any}
    @boundscheck if iterate(x, state) !== nothing
        throw(ArgumentError("Collection has multiple elements, must contain exactly 1 element"))
    end
    return ret
end

@inline function _only(x, ::typeof(first))
    @boundscheck if length(x) != 1
        throw(ArgumentError("Collection must contain exactly 1 element"))
    end
    @inbounds first(x)
end

@propagate_inbounds only(x::IdDict) = _only(x, first)

# Specific error messages for tuples and named tuples
only(x::Tuple{Any}) = x[1]
only(x::Tuple) = throw(
    ArgumentError("Tuple contains $(length(x)) elements, must contain exactly 1 element")
)
only(x::NamedTuple{<:Any, <:Tuple{Any}}) = first(x)
only(x::NamedTuple) = throw(
    ArgumentError("NamedTuple contains $(length(x)) elements, must contain exactly 1 element")
)

"""
    IterableStatePairs(x)

This internal type is returned by [`pairs`](@ref), when the key is the same as
the state of `iterate`. This allows the iterator to determine the key => value
pairs by only calling iterate on the values.

"""
struct IterableStatePairs{T}
    x::T
end

IteratorSize(::Type{<:IterableStatePairs{T}}) where T = IteratorSize(T)
length(x::IterableStatePairs) = length(x.x)
Base.eltype(::Type{IterableStatePairs{T}}) where T = Pair{<:Any, eltype(T)}

function iterate(x::IterableStatePairs, state=first(keys(x.x)))
    it = iterate(x.x, state)
    it === nothing && return nothing
    (state => first(it), last(it))
end

reverse(x::IterableStatePairs) = IterableStatePairs(Iterators.reverse(x.x))
reverse(x::IterableStatePairs{<:Iterators.Reverse}) = IterableStatePairs(x.x.itr)

function iterate(x::IterableStatePairs{<:Iterators.Reverse}, state=last(keys(x.x.itr)))
    it = iterate(x.x, state)
    it === nothing && return nothing
    (state => first(it), last(it))
end

# According to the docs of iterate(::AbstractString), the iteration state must
# be the same as the keys, so this is a valid optimization (see #51631)
pairs(s::AbstractString) = IterableStatePairs(s)
