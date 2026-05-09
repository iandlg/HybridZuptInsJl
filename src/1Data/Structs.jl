struct SeHyperparams
    yaw::Vector{Float64}    # [σ_f, length_scale, σ_n] for yaw
    pos_1::Vector{Float64}  # for x position
    pos_2::Vector{Float64}  # for y position
    pos_3::Vector{Float64}  # for z position

    # Inner constructor to enforce length 3
    function SeHyperparams(yaw, pos_0, pos_1, pos_2)
        for v in (yaw, pos_0, pos_1, pos_2)
            if length(v) != 3
                throw(ArgumentError("Each hyperparameter vector must have exactly 3 elements"))
            end
        end
        new(Vector{Float64}(yaw), Vector{Float64}(pos_0),
            Vector{Float64}(pos_1), Vector{Float64}(pos_2))
    end
end


function SeHyperparams(d::Dict{String,Vector{Any}})
    SeHyperparams(
        Float64.(d["yaw"]),
        Float64.(d["pos_1"]),
        Float64.(d["pos_2"]),
        Float64.(d["pos_3"])
    )
end
