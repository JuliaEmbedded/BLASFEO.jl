# BLASFEO

|                 **License**                 |             **Build Status**              |              **Coverage**              | **Documentation**                                                                           |
|:-------------------------------------------:|:-----------------------------------------:|:--------------------------------------:|---------------------------------------------------------------------------------------------|
| [![License: MIT][license-img]][license-url] | [![build-gh][build-gh-img]][build-gh-url] | [![codecov][codecov-img]][codecov-url] | ![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] |

[license-img]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/JuliaEmbedded/BLASFEO.jl/blob/master/LICENSE
[build-gh-img]: https://github.com/JuliaEmbedded/BLASFEO.jl/actions/workflows/CI.yml/badge.svg
[build-gh-url]: https://github.com/JuliaEmbedded/BLASFEO.jl/actions/workflows/CI.yml
[codecov-img]: https://codecov.io/gh/JuliaEmbedded/BLASFEO.jl/branch/master/graph/badge.svg?token=MBxH2AAu8Z
[codecov-url]: https://codecov.io/gh/JuliaEmbedded/BLASFEO.jl
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliaembedded.github.io/BLASFEO.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://juliaembedded.github.io/BLASFEO.jl/dev/

A Julia package wrapping the [`blasfeo`](https://github.com/giaf/blasfeo) linear algebra library. It is a wrapper around the binary package `blasfeo_jll` which builds architecture specific versions of `blasfeo` for the following hosts:
- `x86_64-[linux|macos|windows]-*` with the following microarchitectures
  - `x86_64`: A fallback option, which does not have platform specific code.
  - `avx`: For old processors supporting only the original avx extension.
  - `avx2`: Modern processors supporting the avx2 extension.
  - `avx512`: For platforms supporting `avx512`. **Warning**: this only provides meaningful speedups for >= workstation class AMD ZEN5 CPUs. For others it may even degrade performance.
- `apple_m1-macos-*` for `>=m1` apple architectures.

Currently other `arm` architectures are unsupported but may be in the future if there are usecases.

## Installation

To install `BLASFEO.jl` using
```julia
pkg> add BLASFEO
```

## Usage
You can start using the BLASFEO API as follows:
```julia
using BLASFEO, LinearAlgebra
A = BlasfeoDmat(rand(100,100))
b = BlasfeoDvec(rand(100))
c = BlasfeoDvec(100)
mul!(c, A, b)
using BLASFEO.LibBlasfeo
d = BlasfeoDvec(100)
blasfeo_dgemv_n(100, 100, 2.0, A, 0, 0, b, 0, 0.0, d, 0, d, 0)

```
