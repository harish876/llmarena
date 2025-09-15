MODEL=$1
COMPS=(
  "aime/aime_2025"
  "hmmt/hmmt_feb_2025"
  "smt/smt_2025"
  "brumo/brumo_2025"
  "cmimc/cmimc_2025"
  
)

for comp in "${COMPS[@]}"; do
  echo "Running on $comp with model $MODEL"
  python scripts/run.py \
    --comp "$comp" \
    --configs "$MODEL" \
    --skip-existing \
    --recompute-tokens
done

python scripts/run.py \
    --comp "apex/apex_2025" \
    --configs "$MODEL" \
    --skip-existing \
    --recompute-tokens \
    --n 16