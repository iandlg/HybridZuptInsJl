"""
The one ordered colour table every figure draws from.

Nine named series appear across the write-up -- three correction methods and six
swept parameter types -- and they have to keep their colour from figure to
figure. Wong has only seven usable entries, which is what forced the previous
arrangement: correction methods on Wong 1-3, hyperparameters on Wong 4-6, and
the normalisation statistics on hand-picked Tol-muted hexes because Wong was
exhausted. Two palettes, two tables, and five plotting files that consulted
neither and coloured by position instead.

`:seaborn_colorblind` is a registered ColorSchemes qualitative scheme with ten
entries, so all nine slots come from one predefined palette in one declared
order, with slot 10 (`#56B4E9`) spare.

| slot | series                              | hex       |
|------|-------------------------------------|-----------|
| 1    | `ZUPT only`                         | `#0173B2` |
| 2    | `Static`                            | `#DE8F05` |
| 3    | `HSGP`                              | `#029E73` |
| 4    | `noise` (σₙ)                        | `#D55E00` |
| 5    | `length_scale` (ℓₛ)                 | `#CC78BC` |
| 6    | `signal_variance` (σ_f)             | `#CA9161` |
| 7    | `input_mean` / `output_mean`        | `#FBAFE4` |
| 8    | `input_std` / `output_std`          | `#949494` |
| 9    | `input_center` / `mid_norm`         | `#ECE133` |

Input and output share a slot per statistic: the colour says *which* statistic,
and input vs output stays readable from the subscript `_hp_type_label` already
draws (μ_x against μ_y). Two same-coloured rows can therefore appear in one
tornado figure, distinguished by their label.

Note that slot 8 is a grey by the palette's own design. It is a real series
colour, not a "no idea" colour -- unknown names get [`FALLBACK_COLOR`], which is
black precisely so that it cannot be mistaken for any palette entry.
"""

const _PALETTE_SCHEME = :tab10

"""
The palette, as `RGBAf`. Ten categorical entries; slots 1-9 are spoken for by
`_SERIES_SLOTS` and 10 is spare (and is where [`series_color_map`](@ref) starts
handing out colours to names it does not know).
"""
const PALETTE = Makie.RGBAf[Makie.RGBAf(Makie.to_color(c))
                            for c in Makie.to_colormap(_PALETTE_SCHEME)]

# Fail at load rather than deep inside a plot call: swapping _PALETTE_SCHEME for
# a scheme with fewer than nine entries is otherwise a BoundsError from whichever
# figure happens to be drawn first.
@assert length(PALETTE) >= 9 "palette $_PALETTE_SCHEME has $(length(PALETTE)) colours, need >= 9"

"""
Colour for a name that is not in [`_SERIES_SLOTS`](@ref).

Black, not grey: slot 8 of the palette *is* a grey, so a grey fallback would be
indistinguishable from a legitimately coloured `input_std` box.
"""
const FALLBACK_COLOR = Makie.RGBAf(0.0, 0.0, 0.0, 1.0)

"""
Series name -> its slot in [`PALETTE`](@ref).

Keyed by the strings the data actually carries: the `estimator` column for the
correction methods, and the `type` column of a `vary_hsgp_parameters` sweep for
the parameter families.

Several series have more than one spelling in the wild, and every spelling of
one series must resolve to its slot -- otherwise the same method is drawn in two
colours within one figure, which is the failure this table exists to prevent:

* The three methods are named `"ZUPT only"` / `"Static"` / `"HSGP"` by the
  `scripts/5Results/` estimator dicts, but the baseline defaults to `"ZUPT INS"`
  in `TrainingDataQualityAnalysis.jl` and `MultiTrackModelTraining.jl`, and the
  `scripts/4OnlineCorrection/` scripts (and the saved CSVs those figures are
  re-plotted from) use `"Decoupled Static"` / `"Decoupled HSGP"` and
  `"Base (no correction)"`.
* The centering family is spelled `input_center` in the sweep's parameter names
  and `mid_norm` in older saved CSVs, after the `HsgpParameters` field it writes
  -- see `_STAT_TYPE_ALIASES` in `Plotting/OnlineHpSensitivity.jl`, which does
  the same for labels.
"""
const _SERIES_SLOTS = Dict{String,Int}(
    "ZUPT only" => 1,
    "ZUPT INS" => 1,
    "Base (no correction)" => 1,
    "Static" => 2,
    "Decoupled Static" => 2,
    "HSGP" => 3,
    "Decoupled HSGP" => 3,
    "noise" => 4,
    "length_scale" => 5,
    "signal_variance" => 6,
    "input_mean" => 7,
    "output_mean" => 7,
    "input_std" => 8,
    "output_std" => 8,
    "input_center" => 9,
    "mid_norm" => 9,
)

"""
    palette_color(slot) -> colour

[`PALETTE`](@ref) entry `slot`, or [`FALLBACK_COLOR`](@ref) if `slot` is outside
the palette. Callers that hold an index rather than a name (the hyperparameter
position in a channel's parameter vector, say) go through here.
"""
palette_color(slot::Int) =
    1 <= slot <= length(PALETTE) ? PALETTE[slot] : FALLBACK_COLOR

"""
    series_color(name) -> colour

Canonical colour for one of the nine named series. Anything else gets
[`FALLBACK_COLOR`](@ref) rather than silently borrowing another series' colour.
"""
series_color(name::AbstractString) =
    palette_color(get(_SERIES_SLOTS, String(name), 0))

"""
    series_color_map(names) -> Dict{String,RGBAf}

Colours for every series in one figure, keyed by name.

Known names take their fixed slot, so a method keeps its colour whether or not
its neighbours are present -- which is the whole point, and the thing that
colouring by position in the figure (or by an `estimator_order` column that
shifts when the baseline is drawn separately) could not do.

Names not in [`_SERIES_SLOTS`](@ref) are dealt the palette slots that no *known*
name present has claimed, starting from the spare slot 10 and wrapping. A figure
carrying an unexpected estimator therefore still gets distinct boxes instead of
several identical black ones; with more unknown names than spare slots the
colours repeat, which is the same failure the positional code always had.
"""
function series_color_map(names)
    keys_ = String.(collect(names))
    taken = Set(get(_SERIES_SLOTS, n, 0) for n in keys_)
    spare = [s for s in Iterators.flatten((10:length(PALETTE), 1:9)) if !(s in taken)]

    out = Dict{String,Makie.RGBAf}()
    next = 1
    for n in keys_
        slot = get(_SERIES_SLOTS, n, 0)
        if slot == 0
            slot = isempty(spare) ? 0 : spare[mod1(next, length(spare))]
            next += 1
        end
        out[n] = palette_color(slot)
    end
    return out
end
