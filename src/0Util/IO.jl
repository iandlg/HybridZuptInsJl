# Source tag (one per file format / data origin)
abstract type AbstractDataSource end
struct ANG <: AbstractDataSource end
struct MTI <: AbstractDataSource end
struct ANG2 <: AbstractDataSource end
struct DCSC <: AbstractDataSource end

# Resolve a directory path to its source tag
const _DIR_TO_SOURCE = Dict{String,AbstractDataSource}(
    "data/angermann_high_precision" => ANG(),
    "data/mti-100-recordings" => MTI(),
    "data/angermann_v2" => ANG2(),
    "data/dcsc_optitrack" => DCSC()
)

function resolve_source(dir::AbstractString)::AbstractDataSource
    src = get(_DIR_TO_SOURCE, dir, nothing)
    isnothing(src) && error("Unknown data directory: $dir\nKnown dirs: $(keys(_DIR_TO_SOURCE))")
    return src
end

function load_hp_variation_results(csv_path::String, json_path::String)::Tuple{DataFrame,Dict}
    df = CSV.read(csv_path, DataFrame)
    metadata = JSON.parsefile(json_path)
    return df, metadata
end

format_vec(v::Vector{Float64}) = "[" * join((@sprintf("%.0e", x) for x in v), ", ") * "]"

function se_vec_str(v::Vector{Float64}, label)
    if length(v) >= 3
        σ_n = v[1]
        σ_f = v[end]
        ℓ = v[2:(end-1)]
        ℓ_str = isempty(ℓ) ? "[]" : "[" * join((@sprintf("%.0e", x) for x in ℓ), ", ") * "]"
        return "$label: σ_n=" * @sprintf("%.0e", σ_n) *
               ", ℓ=" * ℓ_str *
               ", σ_f=" * @sprintf("%.0e", σ_f)
    else
        return "$label: " * format_vec(v)
    end
end
