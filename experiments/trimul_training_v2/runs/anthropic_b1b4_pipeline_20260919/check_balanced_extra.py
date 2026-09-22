"""Extra shape/mask-period audit; preserve the required 24-case result file."""
from check_experiment import *

records = []
with torch.no_grad():
    for length in (72, 80, 136):
        for count in (66, 132):
            for part in (1, 2):
                for dropout in (0.1, 0.25):
                    try:
                        result = run_case('dual_balanced', length, count, part, dropout)
                        result['passed'] = True
                    except AssertionError as exc:
                        result = dict(L=length, count=count, part=part, dropout=dropout,
                                      passed=False, failure=str(exc))
                    records.append(result)
                    (R/'balanced-extra-results.json').write_text(json.dumps(records, indent=2))
    # Repeat the previously documented L72 seed against original and selected.
    d, dy, saved = data(72)
    plans = {name: Experiment(d, dy, saved, 132, 1, name)
             for name in ('dual', 'dual_balanced')}
    diagnosis = []
    for seed in (20261150, 20261151, 20261152):
        change_inputs(d, dy, .25, seed)
        ref = baseline(d, dy, saved)
        outputs = {name: p() for name, p in plans.items()}
        torch.cuda.synchronize()
        diagnosis.append(dict(seed=seed, errors={name:{k:rel(a,b) for k,a,b in zip(NAMES,o,ref)}
                                                for name,o in outputs.items()},
                              dtri_original_selected_bit_exact=torch.equal(
                                  outputs['dual'][2].view(torch.int16),
                                  outputs['dual_balanced'][2].view(torch.int16))))
    (R/'balanced-extra72-results.json').write_text(json.dumps(diagnosis, indent=2))
print('RESULT', sum(r['passed'] for r in records), '/', len(records), flush=True)
