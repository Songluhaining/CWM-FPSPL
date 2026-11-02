"""Freeze after successful TRAIN-only engineering checks, before formal fitting."""
from pathlib import Path
from datetime import datetime
import shutil,sys
from cwm_fp.data.common import ROOT,sha256,write_json
from cwm_fp.models.transition_data import load,verify_source,PREFIX_SHA,PLAN05A_SHA,GRAPH_SHA,VOCAB,hierarchy_weights,read_targets
from cwm_fp.models.graph_encoder import FEATURES,RELATIONS

source=Path(sys.argv[1]);out=Path(sys.argv[2]);manifest,counts,_=verify_source(source)
path=ROOT/'outputs/phase05B_protocol.json'
if path.exists():raise ValueError('Never overwrite frozen protocol')
preflight=load(out/'preflight.json');index=load(out/'cache/index_train.json');targets=read_targets(out,index)
backup=out/'producer_before';backup.mkdir()
for name in ('DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md'): 
    dest=backup/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
files=list((ROOT/'src/cwm_fp/models').glob('transition_*.py'))+[ROOT/'src/cwm_fp/models'/n for n in ('prefix_observer.py','prefix_reader.py','graph_encoder.py')]+[ROOT/'tests/test_transition.py',Path(__file__)]
weights=hierarchy_weights(index['rows'],targets)
p={'stage_id':'05B','protocol':'grouped_bank_pilot_v1','created_at':datetime.now().astimezone().isoformat(),
 'source_manifest_sha256':PREFIX_SHA,'phase05A_protocol_sha256':PLAN05A_SHA,'original_graph_manifest_sha256':GRAPH_SHA,
 'schema':'structural_transition_v1','code_sha256':{x.relative_to(ROOT).as_posix():sha256(x) for x in files},
 'data_counts':counts,'cache_train_index_sha256':sha256(out/'cache/index_train.json'),'known_validation_task_statistics':'05A already public; new fit plan frozen now, not virgin-validation claim',
 'features':list(FEATURES),'input_fields':['x22 semantic/bool','edge_index local','edge_type','observed current anchor local lookup'],
 'excluded':['raw scalar','product or version ID','names/lines','FP/TP targets','future masks','final length/results','target dictionaries'],
 'actual_relations':{str(k):RELATIONS[k] for k in (1,2,3,4,6)},'relation_counts_train':index['stats']['relation_counts_over_prefix_snapshots'],
 'edge_directions':{'control_dep':'static controller->dependent','data_dep':'earlier completed definition event->later completed use event',
 'produces':'completed definition event->observed value version','consumes':'value version->completed use event','temporal':'previous observed product/test event->next observed product/test event'},
 'NoDep_remove_relations':[1,2,3,4],'NoDep_retained':['temporal','self transforms','all nodes; no other input edge families'],
 'model':{'seq_gru':'unidirectional1layer22->32, last valid observed event; Linear32->32 ReLU; A9 B1',
 'rgcn':'existing relation mean encoder2layers32; concatenate anchor32+visible dynamic EVENT mean32+static mean32+3 presence masks ->Linear99->32 ReLU ->A9/B1',
 'no_dep':'identical architecture and initialization to rgcn, fixed4relation removal',
 'static_sharing':'identical disconnected static components computed once per microbatch; independent dynamic blocks; forward/gradient equivalence checked'},
 'parameter_reports':{r['model']:r['parameters'] for r in preflight['records']},'vocabulary_A':list(VOCAB),'B':{'jump':1,'fallthrough':0,'invalid':-1},
 'A_covered':'05A current instruction unique ordinary normal CFG successor exists; independent of later correctness',
 'loss':'(globally hierarchy-normalized Acovered + Aabstain + B)/3; trajectory minibatch estimator scaled by ceil(train_traces/4)',
 'weights':'each part independently: equal valid positions within trajectory, equal eligible trajectories/product, equal eligible products/version, equal eligible versions',
 'weight_sums':weights.sum(0).tolist(),'optimizer':'Adam','learning_rate':.001,'precision':'FP32','extra_class_weights':False,'label_smoothing':False,
 'trajectory_batch':4,'prefix_microbatch':64,'updates_per_epoch':(counts['train']['traces']+3)//4,'epochs':20,'seeds':[20261101,20261102,20261103],
 'candidate_epochs':[5,10,20],'selection':'0.5*version_macro_NLL(Aabstain)/ln9+0.5*version_macro_NLL(B)/ln2; improvement>1e-6 else earlier; missing either ->final20 unselected',
 'version_macro':'within each eligible version event mean then equal versions; not training hierarchical metric',
 'baselines':{'Prior':'TRAIN unsmoothed Aclass frequency, Bjump fraction','Category-Markov':'TRAIN current9category->next9 counts add1; unseencondition->Prior; Bprior',
 'Normal-CFG':'05A exact unique normal successor, abstains; additional static lookup information; retain actual exception errors',
 'Exact-prefix':'per-view TRAIN exact-input target frequency; unmatched->Category-Markov; no new identity features'},
 'duplicate_signatures':'dtype+shape+bytes of used22columns, canonical relation/source/destination edge ordering and anchor; GRU ordered event22 only; excludes scalar and source identity. Exact tensor equivalence, not graph isomorphism/functional equivalence.',
 'subsets':['all','full seen/unseen','own view seen/unseen','three views common unseen'],
 'posthoc_strata':['Acovered','Aabstain','conditional','call','return','actual certified exception','framework interval','both observed branch outcomes at product/domain/BCI point'],
 'metrics':{'A':['accuracy','fixed9macroF1','NLL'],'B':['NLL','Brier','accuracy','balanced accuracy present-class recall','confusion+supports','per product/trace/version n+loss'],
 'NLL_epsilon':1e-12,'B_threshold':.5,'aggregations':['event micro','version macro']},
 'evidence_rules':'Report all seeds; higher ordinary accuracy not nontrivial gain. Evaluate joint dependency family versus NoDep; this single ablation cannot separately identify dynamic-data contribution versus static-control contribution. No causal claim.',
 'device':'cpu','environment_sha256':sha256(out/'environment.json'),'preflight_sha256':sha256(out/'preflight.json'),
 'budgets':{'optimizer_updates_including_engineering':15000,'cache_train_evaluate_seconds':7200,'new_output_bytes':4*2**30,'worker_peak_RSS_bytes':8*2**30,'min_free_bytes':4*2**30},
 'classification_label_access':'prohibited all partitions','test_trajectory_access':'prohibited','new_Java_execution':False,'validation_fit':False,
 'Linux_GPU':'NOT_VERIFIED; no remote session; CPU only','normalization':'per incoming relation mean of currently visible edges; no data-fitted normalizer'}
write_json(path,p);write_json(out/'protocol_receipt.json',{'path':path.relative_to(ROOT).as_posix(),'sha256':sha256(path),'created_at':p['created_at'],'formal_training_started':False})
print('FROZEN',sha256(path),'TRAIN',counts['train'],'preflight_updates',preflight['updates'])
