using Epidemia
using Documenter

DocMeta.setdocmeta!(Epidemia, :DocTestSetup, :(using Epidemia); recursive=true)

makedocs(;
    modules=[Epidemia],
    authors="Andreas Koher",
    repo="https://github.com/andreaskoher/Epidemia.jl/blob/{commit}{path}#{line}",
    sitename="Epidemia.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://andreaskoher.github.io/Epidemia.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/andreaskoher/Epidemia.jl",
)
