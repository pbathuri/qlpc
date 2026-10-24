"""Reproduce all multi-qubit benchmark figures from saved numerical results.

Run: python plot_benchmark.py
Population extrapolation curves have no sampling error bars; their inputs are
exact distributions for the frozen model. RMSE plots include finite shots.
"""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

LABELS = {
    'raw': 'Raw', 'readout': 'Readout correction', 'zne_linear': 'Linear ZNE',
    'readout_zne_linear': 'Readout + linear ZNE',
    'readout_zne_quadratic': 'Readout + quadratic ZNE',
}
COLORS = {'raw':'#60666C','readout':'#0072B2','zne_linear':'#CC79A7',
          'readout_zne_linear':'#009E73','readout_zne_quadratic':'#D55E00'}
MARKERS = {'raw':'o','readout':'s','zne_linear':'^','readout_zne_linear':'D',
           'readout_zne_quadratic':'v'}
OBS = {'mean_z':r'Mean $Z$', 'mean_neighbor_zz':r'Mean adjacent $ZZ$'}
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
    'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
    'axes.spines.top':False,'axes.spines.right':False,
    'savefig.facecolor':'white','figure.facecolor':'white','font.family':'DejaVu Sans'})


def footer(fig, extra=''):
    fig.text(.5,.035,'32,768 shots per protocol, including calibration when used; CZ gate-shots differ.',
             ha='center',fontsize=8,color='#42474B')
    fig.text(.5,.012,'Source: data/benchmark.json (24 Oct 2026). Kingston 15 Apr 2026 calibration in Aer; not live hardware.'+extra,
             ha='center',fontsize=7.7,color='#42474B')


def save(fig, name):
    target = ROOT/'figures'/name
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target,dpi=200)
    plt.close(fig)
    print(target)


def overview(data):
    fig,axes=plt.subplots(1,2,figsize=(11.6,5.3),gridspec_kw={'width_ratios':[1,1.45]})
    ax=axes[0]
    methods=list(data['methods'])
    for y,method in enumerate(methods):
        stats=data['aggregate'][method]
        value=stats['pooled_rmse_ratio_vs_raw']
        lo,hi=stats['pooled_rmse_ratio_bootstrap_95']
        ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt=MARKERS[method],
                    color=COLORS[method],capsize=4,ms=7,lw=1.6)
        ax.text(value+.025,y-.15,f'{value:.3f}',fontsize=9)
    ax.set_yticks(range(len(methods)),[LABELS[m] for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0,1.14)
    ax.set_ylim(4.65,-.65)
    ax.axvline(1,color='#333333',lw=1,ls='--')
    ax.set_xlabel('Pooled RMSE / raw RMSE (dimensionless)')
    ax.set_title('Equal-shot comparison: all 54 case–observables',loc='left',pad=13)
    ax.grid(axis='x',alpha=.2)
    fig.text(.18,.07,'95% paired bootstrap intervals\n(conditional sampling uncertainty)',fontsize=8.5,va='bottom')
    ax=axes[1]
    worse=0
    for case in data['cases']:
        for i,observable in enumerate(case['observables']):
            methods_here=observable['methods']
            ratio=methods_here['readout_zne_quadratic']['rmse_finite']/methods_here['readout_zne_linear']['rmse_finite']
            worse+=ratio>1
            ax.scatter(case['index']+1+(-.12 if i==0 else .12),ratio,
                       color=('#0072B2' if i==0 else '#D55E00'),marker=('o' if i==0 else '^'),
                       s=29,alpha=.85)
    ax.axhline(1,color='#333333',ls='--',lw=1)
    for boundary in (9.5,18.5):
        ax.axvline(boundary,color='#BBBBBB',lw=.7)
    ax.set_xlim(0,28)
    ax.set_ylim(.4,1.82)
    ax.set_xticks([1,5,9,14,19,23,27])
    ax.set_xlabel('Circuit case (fixed grid order; both observables shown)')
    ax.set_ylabel('Quadratic / linear RMSE (dimensionless)')
    ax.set_title(f'With readout correction: quadratic is worse in {worse}/54',loc='left',pad=13)
    for center,n in ((5,4),(14,6),(23,8)):
        ax.text(center,1.75,f'{n} qubits',ha='center',fontsize=9)
    ax.text(27.4,1.025,'worse ↑',ha='right',va='bottom',fontsize=8,color='#555555')
    ax.grid(axis='y',alpha=.2)
    handles=[Line2D([],[],color='#0072B2',marker='o',ls='',label='Mean Z'),
             Line2D([],[],color='#D55E00',marker='^',ls='',label='Mean adjacent ZZ')]
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False)
    fig.suptitle('Noise mitigation lowers pooled error; a higher-order fit often costs accuracy',fontsize=13,y=.965)
    fig.subplots_adjust(left=.18,right=.98,wspace=.37,bottom=.22,top=.83)
    footer(fig)
    save(fig,'mitigation_overview.png')


def depth(data):
    fig,axes=plt.subplots(2,3,figsize=(10.8,6.4),sharex=True,sharey=True)
    for col,n in enumerate((4,6,8)):
        cases=sorted([c for c in data['cases'] if c['n_qubits']==n and c['time']==1.0],key=lambda x:x['steps'])
        for row,observable in enumerate(('mean_z','mean_neighbor_zz')):
            ax=axes[row,col]
            outcomes=[next(o for o in c['observables'] if o['observable']==observable) for c in cases]
            for method in data['methods']:
                ys=[o['methods'][method]['rmse_vs_continuous_finite'] for o in outcomes]
                ax.plot([c['steps'] for c in cases],ys,color=COLORS[method],marker=MARKERS[method],
                        ms=4.5,lw=1.6,label=LABELS[method])
            ax.plot([c['steps'] for c in cases],[abs(o['product_formula_bias']) for o in outcomes],
                    color='#111111',marker='x',ms=5,lw=1.5,ls='--',label='Ideal product-formula |bias|')
            ax.set_yscale('log')
            ax.set_ylim(.0025,.3)
            ax.set_xticks([2,4,8])
            ax.grid(alpha=.2,which='major')
            ax.set_title(f'{n} qubits · {OBS[observable]}',pad=7)
            if col==0:
                ax.set_ylabel('Error vs exact Hamiltonian\n(dimensionless; log scale)')
            if row==1:
                ax.set_xlabel('Strang product-formula steps')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,.075),frameon=False,
               columnspacing=1.6,handlelength=2)
    fig.suptitle(r'Discretization bias and total estimation error: fixed $T=1/J$',fontsize=13,y=.965)
    fig.subplots_adjust(left=.095,right=.98,wspace=.18,hspace=.34,bottom=.26,top=.86)
    footer(fig)
    save(fig,'depth_tradeoff.png')


def folds(data):
    case=next(c for c in data['cases'] if c['n_qubits']==8 and c['time']==1.0 and c['steps']==8)
    factors=np.array(data['fold_factors'])
    grid=np.linspace(0,5,151)
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.8),sharey=True)
    for ax,observable in zip(axes,case['observables']):
        corrected=np.array(observable['fold_population_readout_corrected'])
        ax.plot(factors,observable['fold_population_raw'],'o--',color=COLORS['raw'],lw=1.3,ms=5,label='Raw population mean')
        ax.plot(factors,corrected,'s',color=COLORS['readout'],ms=6,label='Readout-corrected population mean',zorder=5)
        for degree,method,label in [(1,'readout_zne_linear','Fixed linear fit + intercept'),(2,'readout_zne_quadratic','Fixed quadratic fit + intercept')]:
            polynomial=np.polyfit(factors,corrected,degree)
            ax.plot(grid,np.polyval(polynomial,grid),color=COLORS[method],ls=('-' if degree==1 else '-.'),lw=1.6,label=label)
            ax.scatter([0],[np.polyval(polynomial,0)],color=COLORS[method],marker=MARKERS[method],s=42,zorder=6)
        ax.axhline(observable['ideal_digital'],color='#111111',ls='--',lw=1.4,label='Ideal digital reference')
        ax.axhline(observable['cz_noise_off_diagnostic_readout_corrected'],color='#7851A9',ls=':',lw=2,
                   label='CZ-noise-off + readout correction (diagnostic)')
        ax.set_xlim(-.2,5.2)
        ax.set_xticks([0,1,3,5])
        ax.set_xlabel('Native CZ fold factor (dimensionless)')
        ax.set_title(OBS[observable['observable']],pad=10)
        ax.grid(alpha=.17)
    axes[0].set_ylabel('Population expectation (dimensionless)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,.085),frameon=False,
               columnspacing=2.5,handlelength=2.3)
    fig.suptitle(r'Selective extrapolation does not remove all noise: 8 qubits, $T=1/J$, 8 steps',fontsize=13,y=.965)
    fig.text(.5,.896,'Exact model means: no sampling error bars. Diagnostic line is excluded from both fits.',
             ha='center',fontsize=9,color='#42474B')
    fig.subplots_adjust(left=.08,right=.98,wspace=.14,bottom=.33,top=.81)
    footer(fig)
    save(fig,'selective_extrapolation.png')


def main():
    data=json.loads((ROOT/'data/benchmark.json').read_text())
    if len(data['cases'])!=27 or data['total_shots_per_protocol']!=32768:
        raise ValueError('Unexpected benchmark protocol')
    overview(data)
    depth(data)
    folds(data)


if __name__=='__main__':
    main()
