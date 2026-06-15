using Documenter, BLASFEO

makedocs(
    ;
    modules=[BLASFEO],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/JuliaEmbedded/BLASFEO.jl/blob/{commit}{path}#L{line}",
    sitename="BLASFEO.jl",
    authors="Anton Pozharskiy <apozharski@gmail.com>, Ian McInerney <ian.s.mcinerney@ieee.org>",
    format=Documenter.HTML(
        ;
        edit_link="master",
        assets=String[],
    ),
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaEmbedded/BLASFEO.jl",
)
