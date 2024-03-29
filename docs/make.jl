using Documenter, BLASFEO

makedocs(;
    modules=[BLASFEO],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/JuliaEmbedded/BLASFEO.jl/blob/{commit}{path}#L{line}",
    sitename="BLASFEO.jl",
    authors="Ian McInerney, Imperial College London",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaEmbedded/BLASFEO.jl",
)
