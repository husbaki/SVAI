"""Reproduce charts from the saved experiment JSON, never invented metrics."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

source=Path(sys.argv[1] if len(sys.argv)>1 else 'experiments/code')
out=Path(__file__).resolve().parent
q=json.loads((source/'quality_test.json').read_text(encoding='utf-8'))['results']
b=json.loads((source/'benchmark.json').read_text(encoding='utf-8'))['results']
training={v:json.loads((source/(v+'.training.json')).read_text(encoding='utf-8')) for v in ['latent','address','hybrid']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#bcc6c9','axes.labelcolor':'#354852','text.color':'#142b38','xtick.color':'#52636c','ytick.color':'#52636c','figure.facecolor':'#fbfcfa','axes.facecolor':'#fbfcfa','svg.fonttype':'none'})
colors={'none':'#aebbc1','rag':'#448eb0','oracle':'#c5a66b','latent':'#7c81b0','address':'#87b8a2','hybrid':'#247557','hybrid_no_latent':'#91b7aa','hybrid_no_address':'#b2cba3'}
names={'none':'No memory','rag':'Local RAG','oracle':'Oracle context','latent':'A · Latent','address':'B · Address','hybrid':'C · Hybrid','hybrid_no_latent':'Hybrid − latent','hybrid_no_address':'Hybrid − addresses'}
def save(fig,name,subtitle):
    fig.text(.02,.015,subtitle,fontsize=8,color='#78858b')
    fig.tight_layout(rect=(0,.055,1,.98))
    for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

modes=['none','rag','oracle','latent','address','hybrid']
fig,ax=plt.subplots(figsize=(11,5.4));values=[q[m]['exact_match']*100 for m in modes]
bars=ax.barh([names[m] for m in modes],values,color=[colors[m] for m in modes],height=.65)
ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xlabel('Exact generated answer (%)');ax.set_title('Held-out code QA · same frozen Qwen',loc='left',fontweight='bold',pad=20)
for bar,m,v in zip(bars,modes,values):ax.text(v+1,bar.get_y()+bar.get_height()/2,f'{v:.1f}%  ({round(q[m]["exact_match"]*q[m]["n"])}/{q[m]["n"]})',va='center',fontsize=10)
ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
save(fig,'quality_overall','61 held-out questions · 12,244 source candidates · source: quality_test.json. Oracle is a supplied-context control, not learned retrieval.')

repos=['itsdangerous','httpx','django'];modes=['rag','latent','address','hybrid'];fig,axs=plt.subplots(1,3,figsize=(13,5.1),sharey=True)
for ax,repo in zip(axs,repos):
    vals=[(1-q[m][repo+'_error'])*100 for m in modes];bars=ax.bar(np.arange(4),vals,color=[colors[m] for m in modes],width=.65)
    ax.set_xticks(np.arange(4),['RAG','A','B','C']);ax.set_ylim(0,108);ax.set_title(f'{repo}\nn={q["hybrid"][repo+"_n"]}',fontweight='bold');ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
    for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+2,f'{v:.1f}',ha='center',fontsize=9)
axs[0].set_ylabel('Exact generated answer (%)');fig.suptitle('Quality by repository',x=.02,ha='left',fontweight='bold',fontsize=15)
save(fig,'quality_by_repository','Question samples over pinned source snapshots. These scores do not establish correctness of every function or repository behavior.')

modes=['hybrid','hybrid_no_latent','hybrid_no_address'];fig,axs=plt.subplots(1,2,figsize=(12,5.3))
for ax,key,label in [(axs[0],'exact_match','All test questions'),(axs[1],'heldout_behavior_error','Unseen behavior intents')]:
    vals=[(q[m][key] if key=='exact_match' else 1-q[m][key])*100 for m in modes];bars=ax.barh([names[m] for m in modes],vals,color=[colors[m] for m in modes]);ax.invert_yaxis();ax.set_xlim(0,108);ax.set_title(label,fontweight='bold');ax.set_xlabel('Exact generated answer (%)')
    for bar,v in zip(bars,vals):ax.text(v+1,bar.get_y()+bar.get_height()/2,f'{v:.1f}%',va='center')
fig.suptitle('Hybrid channel ablation',x=.02,ha='left',fontweight='bold',fontsize=15)
save(fig,'hybrid_ablation','Same hybrid checkpoint; channels disabled at inference. Unseen behavior track has only 8 executable fixtures; all-test n=61.')

tracks=['behavior','location','heldout_behavior','unknown'];modes=['rag','latent','hybrid'];fig,ax=plt.subplots(figsize=(11,5.3));x=np.arange(len(tracks))
for i,m in enumerate(modes):
    vals=[(1-q[m][t+'_error'])*100 for t in tracks]
    bars=ax.bar(x+(i-1)*.23,vals,.23,label=names[m],color=colors[m])
    for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+1.7,f'{v:.0f}',ha='center',fontsize=9)
ax.set_xticks(x,[f'Behavior\nn={q["hybrid"]["behavior_n"]}',f'Location\nn={q["hybrid"]["location_n"]}',f'Unseen behavior\nn={q["hybrid"]["heldout_behavior_n"]}',f'Unanswerable\nn={q["hybrid"]["unknown_n"]}']);ax.set_ylim(0,110);ax.set_ylabel('Correct response (%)');ax.set_title('Quality by task: aggregate accuracy hides abstention failures',loc='left',fontweight='bold',pad=20);ax.legend(frameon=False,ncol=3,loc='upper left');ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
save(fig,'quality_by_task','Tracks overlap: unseen behavior is a subset of behavior. Unanswerable correctness = 1 − unanswerable error, not source-address validity.')

modes=['rag','latent','address','hybrid'];fig,axs=plt.subplots(1,2,figsize=(12,5.3))
x=np.arange(4);p50=[b[m]['batches']['1']['p50_ms'] for m in modes];p95=[b[m]['batches']['1']['p95_ms'] for m in modes]
axs[0].bar(x-.18,p50,.36,label='p50',color='#438c71');axs[0].bar(x+.18,p95,.36,label='p95',color='#bbcfc4');axs[0].set_xticks(x,['RAG','A','B','C']);axs[0].set_ylabel('End-to-end latency (ms)');axs[0].set_title('Batch = 1 · synchronized wall time');axs[0].legend(frameon=False)
for i,(a,c)in enumerate(zip(p50,p95)):axs[0].text(i-.18,a+35,f'{a:.0f}',ha='center',fontsize=8);axs[0].text(i+.18,c+35,f'{c:.0f}',ha='center',fontsize=8)
v=['address','hybrid','latent'];ts=[training[m]['baking_s'] for m in v];bars=axs[1].barh([names[m] for m in v],ts,color=[colors[m] for m in v]);axs[1].invert_yaxis();axs[1].set_xlim(0,max(ts)*1.22);axs[1].set_xlabel('Baking wall time (seconds)');axs[1].set_title('Trainable memory only · frozen LLM')
for bar,t in zip(bars,ts):axs[1].text(t+1,bar.get_y()+bar.get_height()/2,f'{t:.1f}s',va='center')
save(fig,'latency_and_training','RTX 3050 6 GB · exploratory latency: 7 batches per setting. Training: 103 QA + dev validation; preparation/model loading excluded.')

snapshot=dict(quality=q,benchmark=b,training=training)
(out/'chart_data.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
print('\n'.join(str(p) for p in out.glob('*.png')))
