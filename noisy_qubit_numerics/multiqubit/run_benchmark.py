"""Equal-shot-budget comparison from the frozen device-calibrated distributions.

Trials resample the simulator distributions, not a live quantum device. Fixed
protocols and coupled measurement prefixes are defined in PROTOCOL.md.
"""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from estimators import (FACTORS, METHODS, OBSERVABLES, bit_table, calibration_rates,
                        corrected_weights, estimate_counts, extrapolate, nested_counts,
                        observable_weights, shot_plan)

ROOT=Path(__file__).resolve().parent
REPEATS=500
BOOTSTRAPS=1000
TOTAL_SHOTS=32768
CALIBRATION_SHOTS=2048
SEED=20261024


def finite_summary(values, target):
    finite=np.isfinite(values)
    errors=values[finite]-target
    return {'finite_count':int(finite.sum()), 'undefined_count':int((~finite).sum()),
            'bias_finite':float(errors.mean()) if finite.any() else None,
            'rmse_finite':float(np.sqrt(np.mean(errors**2))) if finite.any() else None,
            'standard_deviation_finite':float(np.std(values[finite],ddof=1)) if finite.sum()>1 else None,
            'outside_physical_range_count':int(np.sum(finite & (np.abs(values)>1)))}


def main():
    metadata=json.loads((ROOT/'data/circuits.json').read_text())
    distributions=np.load(ROOT/'data/distributions.npz',allow_pickle=False)
    plans=shot_plan(TOTAL_SHOTS,CALIBRATION_SHOTS)
    rows=[]; saved={}; squared_errors=[]; continuous_errors=[]; bootstrap_means=[]
    for case in metadata['cases']:
        index=case['index'];n=case['n_qubits'];seed=SEED+index
        rng=np.random.default_rng(seed)
        features=observable_weights(n)
        ideal=distributions[f'case{index}_ideal']@features
        continuous=distributions[f'case{index}_continuous']@features
        cz_off=distributions[f'case{index}_cz_noise_off']@features
        cal0=rng.multinomial(CALIBRATION_SHOTS,distributions[f'cal{n}_zero'],size=REPEATS)
        cal1=rng.multinomial(CALIBRATION_SHOTS,distributions[f'cal{n}_one'],size=REPEATS)
        e0,e1,gap_counts=calibration_rates(cal0,cal1,CALIBRATION_SHOTS,n)
        weights=corrected_weights(e0,e1)
        weights[np.any(gap_counts==0,axis=1)]=np.nan
        bits=bit_table(n)
        population_e0=distributions[f'cal{n}_zero']@bits
        population_e1=1-distributions[f'cal{n}_one']@bits
        population_weights=corrected_weights(population_e0,population_e1)[0]
        fold_counts={}
        for f_index,factor in enumerate(FACTORS):
            budgets=[plans[method][f_index] for method in METHODS]
            fold_counts[factor]=nested_counts(rng,distributions[f'case{index}_fold{factor}'],budgets,REPEATS)
        estimates=np.empty((REPEATS,len(METHODS),len(OBSERVABLES)))
        population={}
        fold_raw=np.array([distributions[f'case{index}_fold{f}']@features for f in FACTORS])
        fold_ro=np.array([distributions[f'case{index}_fold{f}']@population_weights for f in FACTORS])
        for m,method in enumerate(METHODS):
            correct=method.startswith('readout')
            w=weights if correct else features
            if method in ('raw','readout'):
                estimates[:,m]=estimate_counts(fold_counts[1][plans[method][0]],w)
                population[method]=(fold_ro if correct else fold_raw)[0]
            else:
                by_fold=np.array([estimate_counts(fold_counts[f][plans[method][k]],w) for k,f in enumerate(FACTORS)])
                degree=2 if method.endswith('quadratic') else 1
                estimates[:,m]=extrapolate(by_fold,degree)
                population[method]=extrapolate(fold_ro if correct else fold_raw,degree)
        saved[f'case{index}_estimates']=estimates
        saved[f'case{index}_cal_error_counts']=np.stack((e0*CALIBRATION_SHOTS,e1*CALIBRATION_SHOTS),axis=1).astype(np.int64)
        errors=(estimates-ideal)**2
        squared_errors.append(errors)
        continuous_errors.append((estimates-continuous)**2)
        brng=np.random.default_rng(SEED+10000+index)
        # One row index selects every method and observable together.
        samples=brng.integers(0,REPEATS,size=(BOOTSTRAPS,REPEATS))
        boot=np.mean(errors[samples],axis=1)
        bootstrap_means.append(boot)
        case_rows=[]
        for o,observable in enumerate(OBSERVABLES):
            records={}
            for m,method in enumerate(METHODS):
                s=finite_summary(estimates[:,m,o],ideal[o])
                s['rmse_vs_continuous_finite']=finite_summary(estimates[:,m,o],continuous[o])['rmse_finite']
                s['population_estimate']=float(population[method][o])
                s['population_bias_vs_digital']=float(population[method][o]-ideal[o])
                ratio=None;interval=None
                if s['undefined_count']==0 and np.all(np.isfinite(estimates[:,0,o])):
                    ratio=s['rmse_finite']/float(np.sqrt(errors[:,0,o].mean()))
                    interval=np.quantile(np.sqrt(boot[:,m,o]/boot[:,0,o]),[.025,.975]).tolist()
                s['rmse_ratio_vs_raw']=ratio
                s['rmse_ratio_bootstrap_95']=interval
                s['comparisons']={}
                for baseline in ('raw','readout','readout_zne_linear'):
                    bm=METHODS.index(baseline)
                    if s['undefined_count']==0 and np.isfinite(estimates[:,bm,o]).all():
                        denominator=float(np.sqrt(errors[:,bm,o].mean()))
                        s['comparisons'][baseline]={'rmse_ratio':s['rmse_finite']/denominator,
                            'bootstrap_95':np.quantile(np.sqrt(boot[:,m,o]/boot[:,bm,o]),[.025,.975]).tolist()}
                    else:
                        s['comparisons'][baseline]={'rmse_ratio':None,'bootstrap_95':None}
                s['cz_gate_shots']=sum(plans[method][k]*case['folds'][str(f)]['cz'] for k,f in enumerate(FACTORS))
                s['calibration_shots']=2*CALIBRATION_SHOTS if method.startswith('readout') else 0
                s['measurement_shots_by_fold']=plans[method]
                records[method]=s
            case_rows.append({'observable':observable,'ideal_digital':float(ideal[o]),
                              'exact_hamiltonian':float(continuous[o]),
                              'product_formula_bias':float(ideal[o]-continuous[o]),
                              'cz_noise_off_diagnostic_raw':float(cz_off[o]),
                              'cz_noise_off_diagnostic_readout_corrected':float(distributions[f'case{index}_cz_noise_off']@population_weights[:,o]),
                              'fold_population_raw':fold_raw[:,o].tolist(),
                              'fold_population_readout_corrected':fold_ro[:,o].tolist(),
                              'methods':records})
        rows.append({**case,'seed':seed,'zero_gap_replicates':int(np.any(gap_counts==0,axis=1).sum()),
                     'negative_gap_replicates':int(np.any(gap_counts<0,axis=1).sum()),
                     'effective_calibration_e0':population_e0.tolist(),
                     'effective_calibration_e1':population_e1.tolist(),'observables':case_rows})
        print(f'case {index+1}/27: n={n}, t={case["time"]}, steps={case["steps"]}',flush=True)
    errors=np.asarray(squared_errors)
    continuous_errors=np.asarray(continuous_errors)
    boots=np.asarray(bootstrap_means).mean(axis=(0,3))
    aggregate={}
    for m,method in enumerate(METHODS):
        finite=np.isfinite(errors[:,:,m,:])
        complete=bool(finite.all())
        record={'undefined_estimate_count':int((~finite).sum()),
                'pooled_rmse':float(np.sqrt(errors[:,:,m,:].mean())) if complete else None,
                'pooled_rmse_vs_continuous':float(np.sqrt(continuous_errors[:,:,m,:].mean())) if complete else None,
                'case_observables_better_than_raw':sum(o['methods'][method]['rmse_ratio_vs_raw'] is not None and o['methods'][method]['rmse_ratio_vs_raw']<1 for r in rows for o in r['observables']),
                'case_observables_worse_than_raw':sum(o['methods'][method]['rmse_ratio_vs_raw'] is not None and o['methods'][method]['rmse_ratio_vs_raw']>1 for r in rows for o in r['observables']),
                'outside_physical_range_count':sum(o['methods'][method]['outside_physical_range_count'] for r in rows for o in r['observables'])}
        if complete:
            record['pooled_rmse_ratio_vs_raw']=record['pooled_rmse']/float(np.sqrt(errors[:,:,0,:].mean()))
            record['pooled_rmse_ratio_bootstrap_95']=np.quantile(np.sqrt(boots[:,m]/boots[:,0]),[.025,.975]).tolist()
        else:
            record['pooled_rmse_ratio_vs_raw']=None;record['pooled_rmse_ratio_bootstrap_95']=None
        aggregate[method]=record
        record['worse_than_readout_count']=sum(o['methods'][method]['comparisons']['readout']['rmse_ratio'] is not None and o['methods'][method]['comparisons']['readout']['rmse_ratio']>1 for r in rows for o in r['observables'])
        record['worse_than_combined_linear_count']=sum(o['methods'][method]['comparisons']['readout_zne_linear']['rmse_ratio'] is not None and o['methods'][method]['comparisons']['readout_zne_linear']['rmse_ratio']>1 for r in rows for o in r['observables'])
    device=metadata['device']
    result={'date':'2026-10-24','scope':'Aer simulation of a dated device-calibration snapshot; not live hardware',
            'analysis_versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy')},
            'calibration_date':device['calibration_date'],'physical_qubits':device['physical_qubits'],
            'repetitions':REPEATS,'bootstrap_replicates':BOOTSTRAPS,'seed':SEED,
            'total_shots_per_protocol':TOTAL_SHOTS,'calibration_shots_per_circuit':CALIBRATION_SHOTS,
            'shot_plans':plans,'methods':list(METHODS),'observable_names':list(OBSERVABLES),
            'fold_factors':list(FACTORS),'noise_sha256':device['noise_sha256'],
            'source_distribution_sha256':hashlib.sha256((ROOT/'data/distributions.npz').read_bytes()).hexdigest(),
            'bootstrap_scope':'paired rows within fixed cases; conditional Monte Carlo uncertainty only; not simultaneous coverage',
            'aggregate_weighting':'equal weight over27cases x2observables x500replicates; undefined inversions make full aggregate undefined',
            'aggregate':aggregate,'cases':rows}
    (ROOT/'data/benchmark.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(ROOT/'data/trials.npz',**saved)
    with (ROOT/'data/all_results.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case','qubits','time','steps','observable','method','target_digital','target_continuous','population_estimate','bias','rmse','rmse_continuous','ratio_raw','undefined','outside_range','cz_gate_shots'])
        for row in rows:
            for o in row['observables']:
                for method,s in o['methods'].items():
                    writer.writerow([row['index'],row['n_qubits'],row['time'],row['steps'],o['observable'],method,o['ideal_digital'],o['exact_hamiltonian'],s['population_estimate'],s['bias_finite'],s['rmse_finite'],s['rmse_vs_continuous_finite'],s['rmse_ratio_vs_raw'],s['undefined_count'],s['outside_physical_range_count'],s['cz_gate_shots']])
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
