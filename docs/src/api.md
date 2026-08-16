# API
## Data Structures
```@autodocs
Modules = [BLASFEO]
Pages   = ["mats.jl", "vecs.jl"]
```

## "Level 1" API
The level 1 API corresponds to the $O(n)$ operations available over `BLASFEO` objects. We list only the operaions which are currently specialized, other

```@autodocs
Modules = [BLASFEO]
Pages   = ["level1.jl"]
```

## "Level 2" API
```@autodocs
Modules = [BLASFEO]
Pages   = ["level2.jl"]
```
The "level 2" API currently only consists of the in place three argument `LinearAlgebra.mul!` operation for matrix-vector operations, and matrix-vector multiplication via `Base.:*`. These operate as expected on both `BLASFEO` matricies and `Transpose` matricies with `BLASFEO` parents.