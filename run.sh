#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"

case "${1:-}" in
  generate-wilson)       exec bin/generate_wilson ;;
  analyze-wilson)        exec bin/analyze_wilson ;;
  generate-domain-wall)  exec bin/generate_domain_wall ;;
  analyze-domain-wall)   exec bin/analyze_domain_wall ;;
  plot)                  exec python3 scripts/plot_correlators.py ;;
  estimate-mass)         exec python3 scripts/estimate_mass.py ;;
  gevp)                  exec python3 scripts/gevp_spectrum.py ;;
  perambulators)         exec bin/build_perambulators ;;
  contract)              shift; exec python3 scripts/distillation_contract.py "$@" ;;
  fit-correlators)       shift; exec python3 scripts/fit_correlators.py "$@" ;;
  *)
    echo "Usage: $0 {generate-wilson|analyze-wilson|generate-domain-wall|analyze-domain-wall|plot|estimate-mass|gevp|perambulators|contract|fit-correlators}" >&2
    exit 2
    ;;
esac
