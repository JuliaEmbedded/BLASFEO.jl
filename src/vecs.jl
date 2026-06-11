# NOTE(@anton): Both of these structs are treated as mutable and Julia manages their
#               lifetimes itself. Blasfeo also offers a way to preallocate memory but
#               we still need the bitstype struct and to manage that memory so I think
#               may as well use `blasfeo_allocate_*` and `blasfeo_free_*`

# bits clone of panel major `blasfeo_dvec`
mutable struct BlasfeoDvec
	  mem::Ptr{Cdouble} # pointer to passed chunk of memory
	  pa::Ptr{Cdouble} # pointer to a pm array of doubles, the first is aligned to cache line size
	  m::Cint # size
	  pm::Cint # packed size
	  memsize::Cint # size of needed memory

    function BlasfeoDvec(m::Int)
        vec = new(C_NULL,C_NULL,0,0,0)
        @ccall blasfeo.blasfeo_allocate_dvec(m::Cint, pointer_from_objref(vec)::Ptr{BlasfeoDvec})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_dvec(pointer_from_objref(this)::Ptr{BlasfeoDvec})::Cvoid
        end
        return finalizer(destructor, vec)
    end

    function BlasfeoDvec(other::Vector{Cdouble})
        vec = new(C_NULL,C_NULL,0,0,0)
        @ccall blasfeo.blasfeo_allocate_dvec(m::Cint, pointer_from_objref(vec)::Ptr{BlasfeoDvec})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_dvec(pointer_from_objref(this)::Ptr{BlasfeoDvec})::Cvoid
        end

        @ccall blasfeo.blasfeo_pack_dvec(m::Cint,
                                         other::Ptr{Cdouble}, 1::Cint,
                                         pointer_from_objref(vec)::Ptr{BlasfeoDvec},
                                         0::Cint)::Cvoid

        return finalizer(destructor, vec)
    end
end

# bits clone of panel major `blasfeo_svec`
mutable struct BlasfeoSvec
	  mem::Ptr{Cfloat} # pointer to passed chunk of memory
	  pa::Ptr{Cfloat} # pointer to a pm array of floats, the first is aligned to cache line size
	  m::Cint # size
	  pm::Cint # packed size
	  memsize::Cint # size of needed memory

    function BlasfeoSvec(m::Int,n::Int)
        vec = new(C_NULL,C_NULL,0,0,0)
        @ccall blasfeo.blasfeo_allocate_svec(m::Cint, pointer_from_objref(vec)::Ptr{BlasfeoSvec})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_svec(pointer_from_objref(this)::Ptr{BlasfeoSvec})::Cvoid
        end
        return finalizer(destructor, vec)
    end

    function BlasfeoSvec(other::Vector{Cfloat})
        vec = new(C_NULL,C_NULL,0,0,0)
        @ccall blasfeo.blasfeo_allocate_svec(m::Cint, pointer_from_objref(vec)::Ptr{BlasfeoSvec})::Cvoid
        function destructor(this)
            @ccall blasfeo.blasfeo_free_svec(pointer_from_objref(this)::Ptr{BlasfeoSvec})::Cvoid
        end

        @ccall blasfeo.blasfeo_pack_svec(m::Cint,
                                         other::Ptr{Cfloat}, 1::Cint,
                                         pointer_from_objref(vec)::Ptr{BlasfeoSvec},
                                         0::Cint)::Cvoid

        return finalizer(destructor, vec)
    end
end
