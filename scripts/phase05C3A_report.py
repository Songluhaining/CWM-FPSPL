"""Single Chinese handoff, generated inside the container from actual receipts."""
import argparse,csv,json,os,re,resource,shutil,time
from collections import Counter,defaultdict
from pathlib import Path
from cwm_fp.data.common import sha256,write_json
from cwm_fp.models.local_dataset import load

def csvout(p,rows):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    if not Path('/.dockerenv').exists():raise RuntimeError('Container only')
    output=root/'outputs'
    if output.exists():raise ValueError('Never overwrite reports')
    output.mkdir();audit=root/'audit_code_v3';runtime=root/'runtime_code_v1/runtime_readiness.json'
    c=load(audit/'checks.json');rr=load(runtime);fields=load(audit/'field_evidence.json');traces=load(audit/'trace_status.json');protocol=load(root/'transfer/protocol.json')
    commands=[load(p) for p in sorted((root/'command_logs').glob('*.json'))]
    tests=[]
    for command in commands:
        if '-m' in command['command'] and 'unittest' in command['command'] or any('regressions.py' in x for x in command['command']):
            log=next(p for p in (root/'command_logs').glob('*.log') if sha256(p)==command['log_sha256'])
            text=log.read_text();m=re.search(r'Ran (\d+) tests? in',text)
            tests.append({'log':str(log.relative_to(root)),'count':int(m[1]) if m else None,'passed':command['exit_code']==0 and '\nOK' in text,'command':command['command']})
    if len(tests)!=3 or not all(t['passed'] for t in tests):raise AssertionError('Missing test evidence')
    cats=c['first_field_categories'];counter=Counter((f['field_owner'],f['field_name'],f['field_descriptor']) for f in fields)
    fieldtable='\n'.join(f'| `{o}.{n}` | `{d}` | {v} |' for (o,n,d),v in sorted(counter.items()))
    cb=sum(f['callback_present'] for f in fields);no_cb=15-cb
    nextuses=sum(f['first_use'] and f['first_use']['stack_occurrence_link_verified'] for f in fields)
    scoped=Counter(f['scope'] for f in fields)
    missing=[m for m in rr['missing_or_unqualified'] if m.get('user_action_required')]
    packet_flags={
       'FIELD_EVIDENCE_AUDITED':True,'OBSERVED_VALUE_SEPARATED_FROM_WRITE_PROVENANCE':True,
       'LOCAL_OBSERVATION_MODE_VERIFIED':None,'NEW_EMAIL_LOCAL_PAIRS_AVAILABLE':False,
       'JAVA_RUNTIME_READY':rr['JAVA_RUNTIME_READY'],'ORIGINAL_TEST_REPLAY_VERIFIED':False,
       'AGENT_REBUILD_READY':False,'USER_ACTION_REQUIRED':bool(missing),
       'INITIAL_STATE_RECOLLECTED':False,'MODEL_TRAINED':False,'FULL_PRODUCT_ADMISSION_CHANGED':False}
    flag_detail={k:{'value':v,'status':'VERIFIED' if v is True else 'NOT_APPLICABLE_NO_A_CLASS_EVIDENCE' if v is None else 'NOT_EXECUTED' if k in {'ORIGINAL_TEST_REPLAY_VERIFIED','INITIAL_STATE_RECOLLECTED','MODEL_TRAINED'} else 'BLOCKED' if k in {'JAVA_RUNTIME_READY','AGENT_REBUILD_READY'} else 'CONFIRMED_FALSE'} for k,v in packet_flags.items()}
    field_specs=[]
    for (owner,name,desc),n in sorted(counter.items()):
        field_specs.append({'field_owner':owner,'field_name':name,'descriptor':desc,'first_gap_traces':n,
          'needed_observation':'post-successful natural getstatic result + typed callback' if desc=='I' else 'post-successful getfield explicit is_null tag; optional collision-free reference token only if identity comparisons needed',
          'time':'after actual successful field instruction, before first consumer; no access on exception',
          'links':['test instance','thread','call frame','static instruction','dynamic read occurrence','completion sequence'],
          'unknowns_retained':['historical reaching write','entire live heap','effects of untraced library/initialization'],
          'do_not':['reflective early read','force class initialization','invoke business getters for logging','infer from next arithmetic/branch/assertion','merge hashes'],
          'future_pairing':'same test with uninstrumented / frozen agent / new bounded agent; compare complete JUnit count/outcome/exception and order; no implementation in 05C3A'})
    write_json(root/'minimum_observation_spec.json',{'status':'SPECIFICATION_ONLY_NOT_IMPLEMENTED','fields':field_specs,
      'upstream_write_if_later_required':'observe natural clinit, constructor or test reset writes at actual time with event linkage; a read callback alone does not restore history',
      'earliest_blocker_upper_bound_traces':15,'complete_products_recovered':0,'complete_product_recovery_upper_bound':None,
      'upper_bound_reason':'removing first gap does not certify later identity/type/library effects; no complete-product recovery estimate supported'})
    metrics={'raw_traces':24,'raw_instruction_events':c['strict_input_rows_equal'],'product_instruction_events':c['event_counts'].get('product'),
      'test_instruction_events':c['event_counts'].get('test'),'strict_admitted_product_pairs':c['strict_product_admitted'],
      'product_branch_candidates':c['candidate_counts']['product|branch'],'product_arithmetic_candidates':c['candidate_counts']['product|arithmetic'],
      'new_admitted_products':0,'full_product_admission_changed':False,
      'graph_nodes':None,'control_edges':None,'data_edges':None,'unmeasured_reason':'No graph reconstruction/re-admission performed; old graphs frozen'}
    before={**metrics,'field_evidence_categories':None,'category_reason':'previous stage did not certify this A-E distinction'}
    after={**metrics,'field_evidence_categories':cats,'new_observation_conditioned_pairs':0,
       'verified_read_anchors':cats['A'],'conditioned_mode_executed':False,'conditioned_mode_visible_inputs':None,
       'conditioned_mode_valid_targets':None,'not_executed_reason':'A-class gate empty; no new transition producer activated'}
    resources={'command_seconds_before_report':sum(x['seconds'] for x in commands),
      'peak_sampled_main_rss_bytes':max(x['sampled_peak_process_rss_bytes'] for x in commands),
      'peak_self_reported_rss_bytes':max(c['resources']['peak_rss_bytes'],rr['resources']['peak_rss_bytes']),
      'generated_stage_bytes_before_report':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
      'mount_free_bytes':shutil.disk_usage(root).free,'root_free_bytes':shutil.disk_usage('/').free,
      'memory_budget_bytes':4*1024**3,'research_budget_bytes':2*1024**3,'gpu_used':False,'environment_installed':False,
      'monitor':'main-process RSS 0.5s; stage bytes 10s; no hard cgroup limit; short-process peak not reliably sampled',
      'transfer_wall_seconds':None,'total_human_wall_seconds':None,'unmeasured_reason':'no end-to-end transfer/manual timer',
      'failed_commands':[{'command':x['command'],'exit_code':x['exit_code'],'reason':'tuple/list representation comparison in first audit'} for x in commands if x['exit_code']!=0],
      'timeout_oom_truncation':False}
    files_changed=['src/cwm_fp/models/field_evidence.py','tests/test_field_evidence.py','scripts/phase05C1_execute.py',
       *['scripts/phase05C3A_'+n for n in ('prepare.py','runtime.py','audit.py','report.py','regressions.py','run.sh')],
       'DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md']
    split={'grouped':{'train':{'audited_products':2,'independent_versions':1,'new_admitted_products':0,'labels_accessed':False},
       'val':{'products':None,'versions':None,'reason':'not accessed'},'test':{'products':None,'versions':None,'reason':'not accessed'}},
       'leave_out_BankAccountTP':{'status':'NOT_EXECUTED','train':None,'val':None,'test':None},
       'leave_out_Email':{'status':'NOT_EXECUTED','train':None,'val':None,'test':None},
       'neural_training_readiness':'BLOCKED for new Email local pairs; previous Bank results not re-evaluated'}
    unresolved=[{'name':'Missing getstatic read value','traces':no_cb,'root_cause_confirmed':True,'evidence':'audit_code_v3/emitter_evidence.json',
       'repair':'future post-read typed callback, not implemented','whole_product_recovery':None,'state_model_impact':'numeric operands missing'},
       {'name':'Reference hash lacks unambiguous null/reference contract','traces':cb,'root_cause_confirmed':True,
        'evidence':'audit_code_v3/field_evidence.json','repair':'future explicit null bit and separate reference token where required',
        'whole_product_recovery':None,'state_model_impact':'reference/branch value cannot be certified'},
       {'name':'Later library/type/identity gaps and unobserved initialization/reset','traces':None,'root_cause_confirmed':'categories confirmed; full repair benefit not evaluated',
        'evidence':'../phase05C2_v1/verification_code_v3/first_gap_evidence.json','whole_product_recovery':None,
        'state_model_impact':'single field callback cannot restore global trust'},
       {'name':'Linux JDK and lifecycle temp-path preconditions','root_cause_confirmed':True,'evidence':'runtime_code_v1/runtime_readiness.json',
        'whole_product_recovery':0,'state_model_impact':'blocks new Java evidence acquisition; fixing environment is not data recovery'}]
    paths={name:'outputs/phase05C3A_'+name for name in ('protocol.json','review_packet.md','review_packet.json','field_evidence.csv','observation_checks.json','runtime_readiness.json','user_actions.md','product_status.csv','graph_status.csv','summary.md')}
    packet={'stage_id':'05C3A','stage_status':'COMPLETED_EVIDENCE_AUDIT_JAVA_REPLAY_BLOCKED',
      'actual_scope':{'system':'Email','version':'_MultipleBugs_.NOB_1.ID_8','products':2,'traces':24,'first_field_gaps':15},
      **packet_flags,'flag_details':flag_detail,'before_metrics':before,'after_metrics':after,'recovered_graphs':0,'admitted_products':0,
      'rejection_reasons':c['first_field_reasons'],'split_readiness':split,
      'world_model_readiness':{'T1':'PARTIAL inherited strict prefixes; no new graph semantics','T2':'BLOCKED no new qualified local pairs',
        'T3':'NOT_AVAILABLE no interventions','future_isolation_checks':c['physical_suffix_checks']+c['pre_input_suffix_checks']},
      'tests_executed':tests,'real_java_tests_executed':0,'resources':resources,'commands':commands,'changed_files':files_changed,
      'artifact_paths':paths,'evidence_root':str(root),'local_evidence_root':'D:/dataset/whn/cwm_fp_generated/data/processed/local_state/phase05C3A_v1',
      'unresolved_issues':unresolved,'user_actions':missing,'recommended_next_stage':{
        'priority':1,'task':'Qualify Linux Java8 and lifecycle-compatible temporary storage, then separately authorize minimal typed field-read observations',
        'reason':'9 reads lack a numeric callback; 6 reference callbacks are not trustworthy reference values; no graph/network enlargement resolves this',
        'auto_start':False},'full_graph_revalidated':False,'original_source_and_results_preserved':True}
    csvout(output/'phase05C3A_field_evidence.csv',fields)
    prows=[]
    for sel in protocol['selected']:
        tt=[t for t in traces if t['identity'][:3]==sel['identity']]
        ff=[f for f in fields if [f['system_id'],f['bug_version_id'],f['product_id']]==sel['identity']]
        prows.append({**dict(zip(('system_id','bug_version_id','product_id'),sel['identity'])),
           'partition':'train','registered_tests':sel['registered_tests'],'selected_traces':len(tt),'observed_result':sel['observed_result'],
           'new_complete_suite_collected':False,'strict_product_pairs':0,'new_local_pairs':0,'first_field_gap_count':len(ff),
           'field_categories':dict(Counter(f['category'] for f in ff)),'full_product_admission':False,
           'full_graph_revalidation_performed':False,'rejection_codes':'NO_A_CLASS_READ_ANCHOR|INHERITED_FULL_PRODUCT_REJECTED',
           'evidence':'audit_code_v3/field_evidence.json;../phase05C2_v1/outputs/phase05C2_product_status.csv'})
    csvout(output/'phase05C3A_product_status.csv',prows)
    grows=[]
    for t in traces:
        ff=[f for f in fields if f['trace_index']==t['trace_index']]
        grows.append({**dict(zip(('system_id','bug_version_id','product_id','test_id'),t['identity'])),
          'partition':'train','event_count':t['events'],'source_execution_verified':t['source_execution_verified'],
          'strict_barrier_unchanged':True,'field_first_gap':t['field_first_gap'],'field_category':ff[0]['category'] if ff else None,
          'new_local_pairs':0,'full_product_admission':False,'new_graph_revalidation':False,
          'rejection_codes':ff[0]['reason'] if ff else t['first_gap']['reason'] if t['first_gap'] else 'NO_SUPPORTED_PRODUCT_QUERY',
          'evidence':f'audit_code_v3/trace_status.json#{t["trace_index"]};../phase05C2_v1/outputs/phase05C2_graph_status.csv'})
    csvout(output/'phase05C3A_graph_status.csv',grows)
    shutil.copyfile(root/'transfer/protocol.json',output/'phase05C3A_protocol.json')
    shutil.copyfile(runtime,output/'phase05C3A_runtime_readiness.json')
    write_json(output/'phase05C3A_observation_checks.json',{'checks':c,'boundary_evidence':'audit_code_v3/boundary_checks.json',
        'prefix_witnesses':'audit_code_v3/prefix_witnesses.json','minimum_observation_spec':'minimum_observation_spec.json','tests':tests,
        'LOCAL_OBSERVATION_MODE_VERIFIED':None,'reason':'No A-class anchor; visibility classifier tested, transition mode not activated'})
    write_json(output/'phase05C3A_review_packet.json',packet)
    user_actions='''# 阶段05C-3A：用户前置事项

当前需要补齐的是已核验的 **Linux x86_64 Java 8开发工具包（JDK）**，不是SSH、Python、GPU或原始测试数据。

- 依据：原运行使用Windows Oracle `1.8.0_381-b09`，类文件主版本52；原EvoSuite运行库1.0.6及代理摘要已核验并迁移。原Windows可执行文件不能用作Linux运行时。
- 已查找：容器PATH、`/usr/lib/jvm`、`/usr/java`、`/opt/java`、`/opt/jdk`、`/usr/local/java`、`/opt`、`/usr/local`及用户`/workspace`，目录搜索深度限定5；未找到java、javac、jar或Linux JDK包。没有扫描其他用户目录。
- 请提供组织允许使用的Linux Java8发行包及其官方SHA256／可信来源，或指出用户目录内已有安装的准确位置。优先匹配原更新版本；若只能使用其他Java8发行版，需明确替代版本，并在后续验收中单列差异。本次没有擅自下载、选择或安装替代运行时，不能声称已证明兼容。
- 交付目录：`X:/codes/wanghaining_cwm_fp/phase05C3A_v1/environment/incoming/`；容器对应`/workspace/codes/wanghaining_cwm_fp/phase05C3A_v1/environment/incoming/`。也可直接在宿主机个人目录`/sdc/wanghaining/codes/wanghaining_cwm_fp/phase05C3A_v1/environment/incoming/`提供。若X盘不可写，沿用既有SSH文件传输。
- 补齐后可解除“无Linux Java/java编译器”的阻碍；仍须验收原测试生命周期、临时目录和路径兼容性，才能启动最多两测试四进程的配对复现。不会自动恢复模型训练资格。

另一个已确认的工程约束：测试准备代码把`java.io.tmpdir`重设为`/tmp`，容器根分区剩余空间为0。本阶段没有修改测试、挂载或权限。是否实际发生根分区临时写入尚未运行验证；因此这里不提前要求用户清理磁盘或重建容器。后续必须先证明临时文件留在允许的挂载盘；如确需管理员调整，再按明确方案单独处理。

原代理、运行依赖、JUnit入口、产品/测试类和构建源码已自动取得，无需重复提供。未执行任何新增Java测试或新初态采集。
'''
    (output/'phase05C3A_user_actions.md').write_text(user_actions,encoding='utf8')
    seconds=resources['command_seconds_before_report'];peak=resources['peak_self_reported_rss_bytes']/1024**2
    text=f'''# CWM-FP 阶段05C-3A：字段读取证据分层与Java补采前置验收

2026年11月2日。本阶段已完成真实服务器容器内的证据审计、边界验证和运行前置检查。**没有新增合格Email局部转换；Java基线复现因缺少已核验Linux Java8运行时而未执行。** 没有新初态采集、模型训练或完整产品准入变化。

## 1. 直接结论与固定范围

仍是原分组训练区Email版本8的产品`model_m_ca4_0011`和`model_m_ca4_0014`、原24条轨迹；不打开FP/TP、缺陷位置或Email验证／测试日志。原候选分母完整保留：46个产品条件分支、21个产品算术，共67。严格模式（strict）保持历史0/67，3325行旧输入和24条最早拒绝边界经本次重新重放逐项一致。

15条以字段为首个缺口的轨迹：A类{cats['A']}、B类{cats['B']}、C类{cats['C']}、D类{cats['D']}、E类{cats['E']}。**{no_cb}条静态字段读确实没有读值回调；另外{cb}条实例字段读有真实回调，但内容为对象散列格式，不能认证为数值、可靠对象身份或无歧义空值。仅缺历史写入来源、而当前读值已认证的A类为0。** 这不表示六条日志“没有回调”，也不根据后续断言通过反推字段为空。

| 实际字段 | 字节码类型描述符 | 首个缺口轨迹数 |
|---|---|---:|
{fieldtable}

首个字段读作用域：{json.dumps(dict(scoped),ensure_ascii=False)}；产品代码与测试代码分别保存。逐条完整身份、原日志行、静态指令、帧、接收者、回调、时钟和第一次使用见`phase05C3A_field_evidence.csv`及`audit_code_v3/field_evidence.json`。

## 2. 原始证据为何支持这些结论

原执行使用的代理SHA256为`{rr['producer_agent_sha256']}`。本次核对其JAR内{rr['agent_classes_checked']}个类与原构建记录一致，实际构建源码和运行依赖亦按摘要核对；没有用新版本BRFL替换原代理。当前源码中已变化的静态导出辅助文件从03A2冻结备份找回原摘要，首轮缺失记录与补充回执都保留。

`TraceTransformer`对操作码178（读取静态字段）走完成标记分支，因此`GetStaticInst.insertByteCodeAfter`虽然存在，却不是该路径实际调用的值记录入口。原物理日志只有`###RET@opcode=178`及定位，不能从后面的加法结果倒推读取值。类初始化也未形成完整运行写入来源。

操作码180（读取实例字段）按字段描述符插入回调，引用类型经`OpcodeInst.dispatchByDesc`进入对象回调；实际六条均为`Obj:0`。源码将null写为0，也使用`System.identityHashCode`写非空对象，格式没有独立空值标志或无碰撞身份契约。原日志还显示EvoSuite会处理回调类，未重新认证其运行替换效果。因此保留“回调关联可核对，引用值不能认证”的B类结论；不把散列0当实测数值0或凭推测升级为null。

六条实例接收者由真实分配事件和已有栈／局部变量／调用传播定位，而不是用散列合并对象。15条字段读之前的已支持前缀、帧及顺序均单独检查；历史写入来源仍未知，完整堆始终未知。{nextuses}条可以连接到后续第一次实际栈消费，但这只是读取事件到消费位置的联系，不能补齐读取载荷或伪造历史定义使用边。证据：`audit_code_v3/prefix_witnesses.json`、`emitter_evidence.json`。

对24条轨迹的全部字段读取另保存`all_field_reads.csv`，共{c['all_field_reads']}次；其中{c['all_field_raw_scalar_callbacks']}次有格式可解析的原始整数／布尔回调。后续回调位于更早未知语义之后时仍拒绝，不把严格模式遮罩当成“日志根本没有值”，也不借此恢复整段轨迹。类别分布为{json.dumps(c['all_field_categories'],ensure_ascii=False)}。

## 3. 实际修改与新模式是否启动

新增`src/cwm_fp/models/field_evidence.py`：只实现字段证据A—E分层、精确整数／布尔载荷检查及读取完成后的可见性门槛，明确分开观测值、上游写入来源、完整堆、局部转换和完整产品五种状态。新增`tests/test_field_evidence.py`与本阶段准备、运行前置、审计、报告脚本。复用原观察器、解释器、静态身份与05C2来源读取器，未更改这些核心语义。

既有容器命令监测脚本增加可选内存和目录字节上限，本阶段指定4GiB内存、2GiB新目录上限；旧默认内存设置和已冻结脚本保持原含义。没有修改Java源码、代理、原日志、旧标签／划分或图准入。格式契约与当前阶段索引同步更新。

允许的独立观测条件模式应以真实读值完成时刻为起点，保持上游来源和堆未知，沿已认证栈／局部变量传递验证后续使用，在未知调用、类型、异常或身份处停止。**本次A类前置门槛为空，因此没有启用或宣称验证这一转换生产模式。** 只完成证据层及其人工记录边界验证，未用修改布尔标志制造局部样本。

| 指标 | 05C2冻结结果 | 本次结果 |
|---|---:|---:|
| 轨迹／实际指令 | 24／3325 | 24／3325，逐行严格输入一致 |
| 产品／测试指令 | 2194／1131 | 2194／1131 |
| 产品分支／算术候选 | 46／21 | 46／21，不删分母 |
| 严格模式合格转换 | 0／67 | 0／67 |
| 可认证字段读取锚点 | 未作本次分层 | 0／15 |
| 新观测条件模式 | 未实现 | 门槛未满足，未启用 |
| 新模式可见输入／有效目标验证 | 未执行 | 未执行，指标为null |
| 新生成合格转换／完整产品 | 0／0 | 0／0 |
| 覆盖内符号一致率 | 无合格样本 | null，不写100% |

没有新增图节点或依赖边；本阶段没有重建完整图，因此节点／边的新测量为空而非假零。完整分类产品资格原样继承；部分局部读证据即使将来合格，也不能替代完整套件和完整图质量。

## 4. 实际边界验证与失败记录

最终16项新增人工记录测试、47项已有局部状态／前缀回归全部通过，共63项不同测试；连同初版15项专项检查，累计78次测试执行。覆盖读值与历史来源分离、回调前不可见、删除回调、错误帧／接收者／类型、同数值独立读取、散列0拒绝、布尔与整数边界、未知不补零及更早库调用／类型缺口不可恢复。人工A类记录只验证分类器，不能计作真实Email样本或已验证转换模式；未实现新堆状态机，故未声称完整“写入再读取”语义认证。

真实15条首个字段读完成了{c['boundary_checks']}组前／后独立截断、{c['physical_suffix_checks']}次物理观测后缀扰动和{c['pre_input_suffix_checks']}次当前输入后缀扰动，均通过；{c['callback_deletions_checked']}条真实实例回调删除后仍正确拒绝。未来分支、回调、异常／最终记录不改变当前输入；新转换级符号核验无样本，结果为空。全部690份来源文件和旧输入／目标摘要再次核对不变。

首次审计退出1：内存中的方法身份元组与JSON列表直接比较造成误报，尚未生成最终分层结果。修正为规范内容比较后第二次审计成功；第三版进一步把完成标记关联与值回调关联分列，并保留一条原本已有写入来源的读值有效状态。最终在`audit_code_v3`重新核对，首次判断的15条分类和严格准入均未改变；`audit_code_v1`、代码快照与失败日志完整保留。没有因为测试通过而隐藏此次真实执行失败。

## 5. Java复现与代理重建前置条件

真实复用SSH、指定运行中容器`wanghaining_ubuntu_ssh`、镜像`wanghaining_ubuntu_env:20260304`及05C1 Python3.11.13环境。管理员指南复用05C1正文与摘要回执，没有声称重新获取最新网页。所有重放、统计与测试在容器内，宿主机未安装／运行实验环境，GPU未使用。新根为`/workspace/codes/wanghaining_cwm_fp/phase05C3A_v1`，宿主机为`/sdc/wanghaining/codes/wanghaining_cwm_fp/phase05C3A_v1`，X盘为`X:/codes/wanghaining_cwm_fp/phase05C3A_v1`。

原入口是`Phase03TestRunner`经`JUnitCore.run(Request.method(...))`调用一个真实JUnit测试。继承的@BeforeClass、@Before、@After、@AfterClass、规则、类初始化／复位和线程清理均保留，不能直接反射调用测试方法绕过它们。原完整命令、工作目录、classpath、512MiB堆、20,000事件与单次限制见`phase05C3A_runtime_readiness.json`。代表测试确认为产品0011的`Email_ESTest.test20`和`Email_ESTest.test07`，历史两者均完成1个测试、结果通过。

产品／测试类和源码、EvoSuite1.0.6、JUnit所用原JAR、Javassist3.27.0及其他classpath依赖、旧代理、构建源码和配置已核验到位。代理源码闭包齐全不等于能够重建：容器PATH和限定软件／个人目录没有java、javac或jar。原Windows Oracle1.8.0_381-b09不能直接在Linux使用。故`JAVA_RUNTIME_READY=false`、`AGENT_REBUILD_READY=false`；两条代表测试本次**均未执行，Java测试进程数0**，不是复现失败，也不是成功。

还有一项必须继续验收的路径条件：测试准备把`java.io.tmpdir`设为`/tmp`，而根分区可用空间为0。仅设置TMPDIR或启动时属性不能证明其后Java临时写入位于/workspace。本次没有改测试、挂载、权限或清理历史文件；实际写入影响未运行确认，不能伪称已解决。原日志的EvoSuite类复位警告保留；Linux环境继承、编码及资源路径差异亦未被“Python可用”替代。

用户当前需提供的准确项仅见`phase05C3A_user_actions.md`：已核验且允许使用的Linux x86_64 Java8开发包／已有安装位置与来源摘要。其他已找到资源无需重复提供。JDK补齐后仍需完成临时路径及运行器Linux路径适配验收，不能自动启动新插桩或训练。

## 6. 最小必要补采规格与研究决策

下一阶段若另行授权，优先考虑在**自然发生且成功的字段读取完成后**增加精确类型回调：`Client.clientCounter:I`记录实际整数；`Email.subject/body/from`记录独立null标志，只有确需对象同一性时再增加不靠散列的引用标识。绑定真实读取事件、帧、线程和完成顺序，不触发提前类初始化、不反射读静态字段、不调用额外业务getter，不从分支／断言倒推输入。每字段具体规格在`minimum_observation_spec.json`。

历史写入若成为下一任务的必需证据，应另行观察自然初始化、构造或测试复位中的实际写事件；读值回调自身不会补齐这些历史，也不提供完整堆快照。新代理仍须与未插桩／旧代理按相同JUnit生命周期配对，核对结束计数、结果、异常及事件顺序。本阶段没有实施任何新插桩。

解决字段观测缺口最多解除15条轨迹的首个阻碍，这是位置上界，**不是15条完整轨迹或两个产品的恢复承诺**。后续未知类型检查、库效果、静态身份、复位／初始化等仍可能阻断；完整产品恢复上界未证明，机器值为null。当前不满足Email局部训练，也不具备双系统模型泛化证据。T1只保留已有有限可信前缀；T2没有新增合格转换；T3没有真实干预对。不能将读值观测称为因果干预。

唯一优先后续任务：先补齐Linux Java8并验收原生命周期与临时路径，再另行开展最小字段读值观测试采。无需GPU资源，不应先扩大网络或重新训练。

## 7. 执行资源与交付

报告生成前容器命令累计{seconds:.3f}秒，包含首次失败审计；实测进程峰值约{peak:.2f}MiB。阶段目录约{resources['generated_stage_bytes_before_report']/1024**2:.2f}MiB（含迁移包、代码、失败证据），挂载盘剩余约{resources['mount_free_bytes']/1024**3:.2f}GiB。没有超时、OOM或原始日志截断。内存／目录为轮询监测，不是系统硬配额；短命测试进程未必被采到真实峰值。传输和人工总耗时未统一计时，以null标记。报告生成自身另有执行回执。

实际入口为`bash code_v1/scripts/phase05C3A_run.sh runtime code_v1`、`tests code_v1`、`regressions code_v1`、`audit code_v1`及修正版`bash code_v2/scripts/phase05C3A_run.sh audit code_v2`、最终`tests code_v3`和`audit code_v3`。所有精确展开命令、退出码、内存与日志摘要在`command_logs/`；报告脚本对应独立快照执行。复现需新目录，禁止覆盖原产物。

本文件是唯一完整中文研究总结；`summary.md`仅作索引。JSON区分成功、失败、阻塞和未执行；字段CSV含15条完整身份与证据，产品／图CSV覆盖2产品和24测试。完整结果位于本阶段独立目录，报告同步项目outputs并通过X盘核对摘要。未删除历史报告；没有确认可安全删除的重复证据文件。阶段到此结束，不自动执行补采、05C后续阶段或模型训练。
'''
    (output/'phase05C3A_review_packet.md').write_text(text,encoding='utf8')
    (output/'phase05C3A_summary.md').write_text('# 阶段05C-3A总结\n\n完整中文总结：[结果交接报告](phase05C3A_review_packet.md)。\n\n字段证据审计完成；新增局部转换0，Java复现前置条件尚未满足。\n',encoding='utf8')
    # Parse and hash every deliverable after emission, never silently replace originals.
    hashes={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    csv_counts={}
    for f in output.iterdir():
        if f.suffix=='.json':load(f)
        if f.suffix=='.csv':
            with f.open(encoding='utf8',newline='') as h:csv_counts[f.name]=len(list(csv.DictReader(h)))
    if csv_counts['phase05C3A_field_evidence.csv']!=15 or csv_counts['phase05C3A_graph_status.csv']!=24:raise AssertionError('Delivery denominator changed')
    write_json(root/'report_receipt.json',{'files':hashes,'csv_rows':csv_counts,'protocol_unchanged':sha256(output/'phase05C3A_protocol.json')==sha256(root/'transfer/protocol.json'),
       'report_script_sha256':sha256(Path(__file__)),'report_generation_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    print(json.dumps({'flags':packet_flags,'field_categories':cats,'files':len(hashes),'tests':sum(x['count'] for x in tests)},ensure_ascii=False))

if __name__=='__main__':main()
