# BLASFEO
| **License** | **Build Status** | **Coverage** |
|:-----------:|:-----------------:|:----------------:|:------------:|:-------:|
| [![License: MIT][license-img]][license-url] | [![build-gh][build-gh-img]][build-gh-url] | [![codecov][codecov-img]][codecov-url] |
[license-img]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/JuliaEmbedded/BLASFEO.jl/blob/master/LICENSE
[build-gh-img]: https://github.com/JuliaEmbedded/BLASFEO.jl/actions/workflows/CI.yml/badge.svg
[build-gh-url]: https://github.com/JuliaEmbedded/BLASFEO.jl/actions/workflows/CI.yml
[codecov-img]: https://codecov.io/gh/JuliaEmbedded/BLASFEO.jl/branch/master/graph/badge.svg?token=MBxH2AAu8Z
[codecov-url]: https://codecov.io/gh/JuliaEmbedded/BLASFEO.jl

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
pkg> add https://github.com/JuliaEmbedded/BLASFEO.jl
```

## Usage
TODO: add examples