push!(LOAD_PATH,"../src/")
using Documenter, BLASFEO

makedocs(
    ;
    modules=[BLASFEO],
    pages = [
    "Home" => "index.md",
    "API" => "api.md",
    "C API" => "c_api.md",
    ],
    repo="https://github.com/JuliaEmbedded/BLASFEO.jl/blob/{commit}{path}#L{line}",
    sitename="BLASFEO.jl",
    authors="Anton Pozharskiy <apozharski@gmail.com>, Ian McInerney <ian.s.mcinerney@ieee.org>",
    format=Documenter.HTML(
        ;
        edit_link="master",
        assets=String[],
        prettyurls = get(ENV, "CI", nothing) == "true",
        size_threshold_ignore = ["c_api.md"],
    ),
)

deploydocs(
    ;
    repo="github.com/JuliaEmbedded/BLASFEO.jl",
    push_preview = true,
    devbranch = "master",
)
