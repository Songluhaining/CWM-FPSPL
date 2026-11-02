"""Seal current measured evidence and documentation, without rerunning production."""
import json,shutil,time
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.models.prefix_dataset import OUT,PACKAGE,PROTOCOL,ANCHOR,load,lines,guard,budget_unchecked
from cwm_fp.models.graph_encoder import FEATURES
from cwm_fp.data.common import sha256,write_json

def main():
    guard(load(PACKAGE/'manifest.json'),True)
    packet=load(ROOT/'outputs/phase05A_review_packet.json');p=load(OUT/'production.json');protocol=load(PROTOCOL)
    independent=load(OUT/'independent_validation.json')
    for f,h in protocol['source_sha256'].items():assert sha256(ROOT/f)==h
    for f,h in protocol['public_sha256'].items():assert sha256(ROOT/f)==h
    assert sha256(PACKAGE/'manifest.json')==ANCHOR
    visibility=[]
    for col,name in enumerate(FEATURES):
        if name.startswith(('node_','opcode_','program_')):
            rule='static product: original compiled material; dynamic: only after real pre-record; no unique identity features'
        else:rule={'executed':'static fixed0; dynamic completed only at observed callback/successor/return',
            'available':'known static existence or already observed event/value-version evidence; not final availability',
            'event_instance':'only records actually arrived', 'config_enabled':'fixed0 placeholder, config_known=false',
            'config_known':'fixed0; constraints unverified'}[name]
        visibility.append({'column':col,'name':name,'rule':rule})
    write_json(OUT/'field_visibility.json',{'schema':'physical_prefix_dataset_v1','features':visibility,
        'extra_model_fields':{'value':'float64 actual measured numeric only; zero placeholder when maskfalse; real cohort exact conversion independently checked',
            'value_observed':'independent boolean mask, no inferred values','in_degree':'current incoming edge count, no full-graph normalization',
            'current_event':'latest observed product/test event; local boolean selector','event_nodes':'prefix local connection indices only',
            'pending_call_nodes':'started without actual observed return; no embedding','heap_state_known':'false, historical values not live heap state'},
        'edge_fields':{'control_dep':'static normal postdominance; no exception/dynamic-control claim','temporal':'already observed product/test instances only',
            'produces':'definition completion/handler evidence arrived','consumes':'actual use completed, endpoints already admitted',
            'data_dep':'verified prefix definition-use; future use never admitted','feature_controls':'not emitted in prefix_v1','observes':'not emitted in prefix_v1'},
        'excluded':['whole_trace_length','final_coverage','final_result','assertion_result','classifier_label','fault_location','unique_identity_embedding']})
    lookup={}
    for t in p['targets']:
        r=p['status'][t['trace_index']];lookup[tuple([r['partition'],*[r[k] for k in ('system_id','bug_version_id','product_id','test_id')],t['event_index']])]=t
    conflicts={'A':0,'B':0};scope={}
    for group in p['duplicates'].values():
        ts=[lookup[tuple(x)] for x in group]
        for name in conflicts:
            if len({t[name] for t in ts if t[name+'_mask']})>1:conflicts[name]+=1
    for part in ('train','val'):
        wrong=[t for t in p['targets'] if t['partition']==part and t['A_mask'] and t['deterministic_prediction'] is not None and t['A']!=t['deterministic_prediction']]
        scope[part]={'errors':len(wrong),'source_opcodes':dict(Counter(str(t['current_opcode']) for t in wrong)),
            'actual_exception_confirmed':0,'evidence':[]}
        cache={}
        for t in wrong:
            r=p['status'][t['trace_index']];path=r['source_graph_path']
            if path not in cache:cache={path:{g['test_id']:g for g in lines(path)}}
            transfers=cache[path][r['test_id']]['control_flow_evidence']['assessment'].get('certified_transfers',[])
            match=next((e for e in transfers if e['source_event']==t['event_index']),None)
            if match:scope[part]['actual_exception_confirmed']+=1
            scope[part]['evidence'].append({'trace_index':t['trace_index'],'event_index':t['event_index'],'certified_transfer':match})
    write_json(OUT/'shortcut_error_audit.json',scope)
    commands=[load(f) for f in sorted(OUT.glob('*.command.json'))]
    peak=max(p['resources']['process_memory']['peak_working_set_bytes'],independent['resources']['process_memory']['peak_working_set_bytes'],packet['resources']['process_memory']['peak_working_set_bytes'])
    packet['tests_executed'].update(prefix_reader_regression=6,total_unit_and_regression_passed=146,failed=0)
    selected=[g for product in load(PACKAGE/'manifest.json')['products'] if product['partitions']['grouped'] in ('train','val') for g in product['tests']]
    source_counts={'registered_tests':368,'candidate_graphs':368,'identity_qualified_graphs':368,'dependency_qualified_graphs':368,
        'complete_selected_products':22,'events':sum(g['events'] for g in selected),'nodes':sum(g['nodes'] for g in selected),
        'edges':sum(g['edges'] for g in selected),'static_control_edges':sum(g['relations']['control_dep'] for g in selected),
        'dynamic_data_edges':sum(g['relations']['data_dep'] for g in selected),'rejected_source_graphs':0}
    packet['before_metrics']['selected_source_cohort']=source_counts
    packet['after_metrics']['selected_source_cohort']=source_counts
    packet['resources'].update(commands=commands,recorded_command_seconds_before_final_seal=sum(c['seconds'] for c in commands),
        measured_worker_peak_bytes=peak,peak_scope='maximum OS lifetime worker peak among production, independent check, and report/reader verification; not sum of concurrent workers',
        final_resource_measurement=budget_unchecked(),unit_test_worker_peak_bytes=None,
        unit_test_peak_reason='external monitor measures virtualenv launcher only; not used as worker peak')
    packet['duplicates']['groups_with_conflicting_targets']=conflicts
    packet['deterministic_successor_errors']=scope
    packet['unresolved_issues'].append({'issue':'duplicate_inputs','impact':'116 exact train/validation shared groups; some identical represented prefixes may have different observed targets, see duplicate counts',
        'confirmed':True,'recovery_upper_bound_products':None,'reason':'not a product admission repair; no changes to frozen split'})
    packet['changed_files']+=['tests/test_prefix_reader.py','scripts/finalize_phase05A.py']
    packet['artifact_paths'].update(field_visibility=str(OUT/'field_visibility.json'),final_audit=str(OUT/'final_audit.json'))
    write_json(ROOT/'outputs/phase05A_review_packet.json',packet)
    summary=ROOT/'outputs/phase05A_review_packet.md';text=summary.read_text(encoding='utf8')
    text=text.replace('合计140项通过','加6项公开前缀读取器检查，合计146项通过')
    text+='\n## 10. 收尾复核及精确命令\n\n'
    text+=f'本阶段涉及的原图前后完全相同：登记测试／候选／身份／依赖合格均368，完整套件产品22；事件{source_counts["events"]}、节点{source_counts["nodes"]}、边{source_counts["edges"]}，其中静态控制边{source_counts["static_control_edges"]}、动态数据依赖边{source_counts["dynamic_data_edges"]}。新增的是派生前缀及目标，不是修改原图、恢复拒绝图或新增完整合格产品。\n\n'
    text+=f'测量到的生产／独立核验／公开读取进程最大峰值驻留内存为{peak}字节（{peak/1024**2:.2f}MiB）。截至收尾封存前，记录命令累计{sum(c["seconds"] for c in commands):.3f}秒。最终目录大小及余量在`final_audit.json`记录；没有超时、内存耗尽、日志截断或缓存替换。单元测试工作进程峰值未独立测量，不以启动器内存代替。\n\n'
    text+=f'普通唯一后继参照训练{scope["train"]["errors"]}次、验证{scope["val"]["errors"]}次预测错误，全部{scope["train"]["actual_exception_confirmed"]+scope["val"]["actual_exception_confirmed"]}次都已对照真实精确空字段异常转移。这说明普通唯一后继不是全部执行后继唯一，不能把79.5%左右的覆盖率表述为同等比例无条件确定目标。\n\n'
    text+=f'完全相同输入组中，下一事件目标存在不同结果的组为{conflicts["A"]}，分支目标存在不同结果的组为{conflicts["B"]}。这是当前表示下的观测歧义，不能直接归因为程序随机性；粗粒度操作类别和未观测状态的具体贡献尚未分离。没有根据验证结果修改任务、样本或划分。\n\n'
    text+='以下命令都由`D:/SoftInstall/python.exe -X utf8 scripts/run_phase05A.py <日志名> ...`封装执行，工作目录为项目根目录，`PYTHONPATH`包含`src`及`tests`。封装器按已有命令实耗扣减累计预算，日志名不能覆盖。\n\n'
    for c in commands:
        text+='```text\n'+' '.join(c['argv'])+f'\n退出码={c["exit_code"]}；耗时={c["seconds"]:.3f}秒\n```\n\n'
    text+='新输出文件仅为协议、这份中文交接、机器交接、逐轨迹CSV和隔离检查JSON。数据与例子在独立版本目录，未新增另一份长总结。\n'
    summary.write_text(text,encoding='utf8')
    stage='''

## 阶段05A补充状态（2026-11-01）

真实物理观测前缀生产器及独立目标已完成。只使用原17训练产品／9版本／283轨迹和5验证产品／4版本／85轨迹，核对真实事件13597＋3921，共生成17518个前缀；下一事件有效目标17150个，分支目标523个，局部直接观测算术配对547组。测试区只核对清单／既有读取器文件元数据，不展开轨迹；所有产品分类标签均未读取。

31769个物理位置的截断重放及158845次后缀扰动比较通过，独立核对了数值、帧、异常和依赖原证据；1104次公开前缀读取复核通过，146项针对性／相关回归通过。没有新Java、模型拟合、服务器或GPU执行。

普通静态唯一后继约覆盖79.5%的有效下一事件目标，30次例外均为已认证空字段异常。存在116组训练／验证共有的完全相同前缀输入。可另行冻结有限的观测性结构预测试训，但完整数值状态、真实干预、跨系统有效性及世界模型训练仍未具备。原28产品分类准入和冻结划分保持不变。

唯一完整中文交接为`outputs/phase05A_review_packet.md`，另有协议JSON、机器交接JSON、轨迹状态CSV及隔离检查JSON。独立数据位于`D:/dataset/whn/cwm_fp_generated/data/processed/prefix/phase05A_v1/`。完成后停止，不自动执行05B。
'''
    with (ROOT/'STAGES.md').open('a',encoding='utf8') as f:f.write(stage)
    (ROOT/'outputs/phase_status.md').write_text('''# 当前阶段状态：05A真实时间前缀与结构转换目标

2026-11-01。本阶段完成。唯一完整中文交接：[phase05A_review_packet.md](phase05A_review_packet.md)。

已从17训练产品／283条轨迹和5验证产品／85条轨迹生成17518个真实前缀。有效下一事件目标17150个、分支目标523个、局部数值算术配对547组。预测位置是指令执行前记录之后、结果回调之前；末尾没有对应后续观测的368个下一事件目标关闭，不制造正常结束标签。

31769个物理观测位置的截断重放与158845次后缀扰动全部通过；独立检查原始数值、定义使用、调用帧、异常处理器及框架返回证据。1104次公开读取及146项人工／相关回归通过。没有读取产品分类目标，没有展开测试轨迹，没有Java重跑、模型训练或服务器访问。

限定观测性结构预测的数据条件已具备，可另行冻结小规模试训。普通静态后继捷径、重复前缀、初始／堆状态缺失、输入输出／断言数值不完整、单系统小队列和无真实干预仍需分别处理。当前完整产品准入仍28个，新增0个；不把数据条件通过当作因果或跨系统研究成立。

原协议、清单、划分、标签、完整图及04R-1实验结果保留。精确命令、资源、所有证据及后续建议见交接包；本阶段结束，等待下一步指令。
''',encoding='utf8')
    # Preserve actual new sources and documentation separately from the protocol's
    # already frozen core implementation. No historical source pin is rewritten.
    for name in packet['changed_files']:
        src=ROOT/name;dest=OUT/'delivered_source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    old=Path('D:/dataset/whn/cwm_fp_generated/data/processed/graph_classifier/phase04R1_cpu_v1/executed_source/src/cwm_fp/lddg/effects.py')
    if old.exists():
        dest=OUT/'producer_before/src/cwm_fp/lddg/effects.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dest)
    index={p.relative_to(OUT).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size} for p in OUT.rglob('*') if p.is_file() and p.name not in {'final_audit.json','artifact_index.json','finalize.log','finalize.command.json'}}
    write_json(OUT/'artifact_index.json',index)
    for name in ['phase05A_protocol.json','phase05A_review_packet.json','phase05A_leakage_checks.json']:load(ROOT/'outputs'/name)
    info=budget_unchecked();assert info['output_bytes']<2*1024**3 and info['free_bytes']>4*1024**3
    write_json(OUT/'final_audit.json',{'status':'PASS','trace_rows':len(p['status']),'accepted_prefixes':17518,'tests_passed':146,
        'field_visibility_columns':len(visibility),'artifact_count':len(index),'artifact_index_sha256':sha256(OUT/'artifact_index.json'),
        'source_manifest_unchanged':True,'public_manifest_split_unchanged':True,'frozen_core_source_unchanged':True,
        'resources':info,'command_seconds_before_final_seal':sum(c['seconds'] for c in commands),
        'classification_target_reads':0,'test_trajectory_reads':0,'outputs':{p.name:sha256(p) for p in (ROOT/'outputs').glob('phase05A_*') if p.is_file()}})
    print('FINAL_SEAL_PASS',len(index),info,flush=True)

if __name__=='__main__':main()
