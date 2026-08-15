# BLASFEO.jl
This package is a Julia wrapper for the [BLASFEO](https://blasfeo.syscop.de/overview/) library, which provides a set of basic linear algebra routines, performance-optimized for matrices of moderate size (up to a couple hundreds elements in each dimension), as typically encountered in embedded optimization applications. It was originally written by and is maintained by Gianluca Frison.
Currently this entails a thin wrapper around the [BLASFEO API](https://blasfeo.syscop.de/docs/api/blasfeoapi/).
We currently further plan to provide a higher level interface for users to be able to take advantage of the additional throughput that BLASFEO provides without resorting to the C interface.

## Contents

```@contents
```