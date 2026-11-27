#!/usr/bin/env bash
# Collect the two files that go to Canvas into submission/:
#   F21BC_Coursework_Report.pdf   the report
#   F21BC_Coursework_Code.zip     the code, data, tests and results
# Run from the repository root:  bash make_submission.sh
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p submission
rm -f submission/F21BC_Coursework_Code.zip

# The code archive: everything needed to run and to check the results,
# without caches or the report sources.
zip -qr submission/F21BC_Coursework_Code.zip \
    README.md requirements.txt main.py run_experiments.py analyse_results.py \
    src experiments tests data results \
    -x '*/__pycache__/*' -x '*.pyc' -x '*/.pytest_cache/*' -x '*.DS_Store'

cp report/F21BC_Coursework_Report.pdf submission/
cp report/F21BC_Coursework_Report.docx submission/

echo "submission/ now contains:"
ls -lh submission
