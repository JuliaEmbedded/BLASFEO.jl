macro checkvvdims(x, y)
    return esc(
    quote
        @boundscheck begin
            (length($x) == length($y)) || throw(DimensionMismatch("Mismatched vector dimensions"))
        end
    end
    )
end

macro checkvvdims(x,y,z)
    return esc(
    quote
        @boundscheck begin
            ((length($x) == length($y)) && (length($x) == length($z))) || throw(DimensionMismatch("Mismatched vector dimensions"))
        end
    end
    )
end
