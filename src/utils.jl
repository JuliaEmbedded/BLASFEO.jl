macro checkvecdims(x,y)
    return esc(
    quote
        @boundscheck begin
            (length($x) == length($y)) || throw(DimensionMismatch("Mismatched vector dimensions"))
        end
    end
    )
end
