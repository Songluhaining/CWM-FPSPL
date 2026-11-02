"""Build one Chinese handoff and machine tables from executed container evidence."""
import argparse,csv,hashlib,json,re,shutil,statistics,time
from collections import Counter,defaultdict
from pathlib import Path

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(cols,rows):return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def csvwrite(p,rows):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;out=r/'outputs'
    if out.exists():raise ValueError('Do not overwrite a delivered report')
    out.mkdir();d=r/'diagnostic_code_v4';s=load(d/'semantic_checks.json');ind=load(r/'final_checks/independent_checks.json');env=load(r/'final_checks/environment.json')
    first=load(r/'gpu_code_v3/server_acceptance.json');last=load(r/'gpu_code_v4/server_acceptance.json');old=load(r/'old_gpu_code_v3/linux_acceptance.json')
    models=[first['records'][0],*last['records']];commands=[load(p) for p in sorted((r/'command_logs').glob('*.json'))]
    raw=[json.loads(line) for part in ('train','val') for line in (d/f'{part}_local_state.jsonl').read_text().splitlines()]
    graphs=load(d/'graph_status.json');bytrace=defaultdict(list)
    for row in raw:bytrace[row['audit']['trace_index']].append(row)
    graph_rows=[];products=defaultdict(list)
    for i,g in enumerate(graphs):
        rr=bytrace[i];bad=[x for x in rr if x['prediction'] is None];identity=g['identity'];products[tuple(identity[:3])].append(g)
        graph_rows.append(dict(zip(('system_id','bug_version_id','product_id','test_id'),identity),partition=g['partition'],trace_index=i,
            event_count=g['events'],identity_verified_count=g['identity_verified'],branch_targets=g['branch_rows'],arithmetic_pairs=g['arithmetic_rows'],
            symbolic_checked=sum(x['prediction'] is not None for x in rr),symbolic_inconsistent=sum(x['consistent'] is False for x in rr),
            diagnostic_abstention_count=len(bad),reason_codes='|'.join(sorted({'RETURN_COPY_NOT_DIRECT_OBSERVATION' if x['task']=='branch' else 'I2L_SEMANTICS_NOT_CERTIFIED' for x in bad})),
            graph_status='FROZEN_INPUT_UNCHANGED;NO_NEW_GRAPH_ADMISSION',evidence=f'diagnostic_code_v4/{g["partition"]}_local_state.jsonl;trace={i};'+g['evidence']))
    product_rows=[]
    for identity,gs in products.items():
        product_rows.append(dict(zip(('system_id','bug_version_id','product_id'),identity),partition=gs[0]['partition'],frozen_selected_test_count=len(gs),
            audited_test_count=len(gs),events=sum(g['events'] for g in gs),identity_verified_events=sum(g['identity_verified'] for g in gs),
            original_suite_completeness='INHERITED;NOT_RECOLLECTED_OR_REAUDITED',original_registered_test_count='',
            missing_count_reason='05C1 uses frozen05A trajectory scope; original suite registry not independently reopened',
            transition_input_readiness='READY_FOR_BOUNDED_OBSERVATIONAL_TASK',FP_TP_training_admission='NOT_REEVALUATED;PRODUCT_TARGETS_PROHIBITED',
            evidence='snapshot/data/05A/prefix_manifest.json;final_checks/independent_checks.json'))
    csvwrite(out/'phase05C1_graph_status.csv',graph_rows);csvwrite(out/'phase05C1_product_status.csv',product_rows)
    shutil.copyfile(d/'local_state_status.csv',out/'phase05C1_local_state_status.csv');write(out/'phase05C1_semantic_checks.json',dict(s,independent_source_audit=ind['state_source_audit']))
    flags={k:True for k in ['GUIDE_READ','SSH_CONNECTED','X_HOST_CONTAINER_MAPPING_VERIFIED','HOST_ENVIRONMENT_UNCHANGED','CONTAINER_ENVIRONMENT_READY','GPU_FORWARD_BACKWARD_VERIFIED','TRAINING_SMOKE_COMPLETED','LOCAL_OPERAND_DATA_GENERATED','SYMBOLIC_CONSISTENCY_VERIFIED','STATIC_DYNAMIC_IDENTITY_LINK_AUDITED']}
    flags.update(CROSS_SYSTEM_EVALUATED=False,CAUSAL_INTERVENTION_VALIDATED=False)
    paths={'WINDOWS_TRANSFER_ROOT':'X:/codes/wanghaining_cwm_fp/phase05C1_v1',
        'HOST_PROJECT_ROOT':'/sdc/wanghaining/codes/wanghaining_cwm_fp/phase05C1_v1','CONTAINER_PROJECT_ROOT':str(r),
        'CONTAINER_DATA_ROOT':str(r/'snapshot/data'),'CONTAINER_RUN_ROOT':str(r),'working_code_snapshots':[str(r/'code_v3'),str(r/'code_v4')],
        'local_evidence_root':'D:/dataset/whn/cwm_fp_generated/data/processed/local_state/phase05C1_v1/remote_evidence'}
    acceptance={'stage_id':'05C1','status':'PASSED_WITH_BYTE_EXACT_OUTPUT_CAVEAT','flags':flags,'paths':paths,
        'server':'222.201.187.68','ssh_port':22,'authentication':'existing key, StrictHostKeyChecking=yes; supplied password never used or stored',
        'host':'2080TI','container_name':'wanghaining_ubuntu_ssh','container_id':'81e96c2ae461e8a5acbe7820d5dd36eb0539a367ba51c31368c7446490395473',
        'image':'wanghaining_ubuntu_env:20260304','host_gpu_index':1,'torch_visible_device':'cuda:0','gpu_uuid':'GPU-e611ba0c-db46-7d8d-486c-3fd79962b2ac',
        'environment':env,'models':models,'total_optimizer_updates':226,'full_epoch_updates':213,'engineering_updates':13,
        'epochs_per_model':1,'seed':20261101,'formal_comparison':False,'validation_checkpoint_selection':False,
        'prior_failures':[{'file':'gpu_code_v3/server_acceptance.json','reason':'extra byte-exact output check failed on no_dep; original logits tolerance unchanged','max_abs_difference':1.9073486328125e-6}],
        'original_tolerances':first['tolerances'],'real_gpu_trajectory_isolation':ind['gpu_real_batch_isolation'],
        'root_filesystem_full':True,'workaround':'all environments, caches, data and runs under the existing /workspace bind mount; no host permission or software changes',
        'x_write_probe':'FAILED_WITH_EMPTY_FILE_RETAINED; SSH transfer succeeded; X read/hash verified',
        'path_probe_sha256':'542f314456601fc2ba9cff48630f224f09aaf7ba349ba3d4820d4e6b3f67c16a',
        'guide_receipt':load(r/'receipt.json'),'guide_embedded_inventory':'NOT_READ; separate linked document organization_limited; live machine inspection used',
        'peak_torch_allocated_bytes':max([old['peak_allocated_bytes']]+[x['peak_allocated_bytes'] for x in models]),
        'peak_torch_reserved_bytes':max([old['peak_reserved_bytes']]+[x['peak_reserved_bytes'] for x in models])}
    write(out/'phase05C1_server_acceptance.json',acceptance)
    changed=['src/cwm_fp/models/local_state.py','tests/test_local_state.py','tests/test_prefix_reader.py',
        *['scripts/phase05C1_'+n for n in ['export.py','execute.py','run.sh','diagnose.py','gpu_acceptance.py','finalize_checks.py','report.py']],
        'DATA_CONTRACTS.md','STAGES.md','outputs/phase_status.md']
    issue_specs=[
        ('CROSS_SYSTEM_UNVALIDATED','仅BankAccountTP的9训练/4验证版本；没有Email训练或跨系统评价。',22,368,'snapshot/data/05A/prefix_manifest.json','首要建议为有限Email机制数据接入；需新阶段授权。'),
        ('RETURN_COPY_NOT_DIRECT','105个分支的调用返回值有复制证据，但没有返回版本自己的直接标量回调；不纳入主核验。',12,102,'final_checks/independent_checks.json','复制链已可靠追溯；若未来必须直接观测，需要单独授权最小返回回调采集。'),
        ('I2L_NOT_CERTIFIED','75组int转long配对输入真实可见，但本轮未认证宽数值转换语义。',None,None,'diagnostic_code_v4/local_state_status.csv','可作为下一阶段有限语义扩展候选；本轮不补实现。'),
        ('QUERY_VALUE_DISCONNECTED','1070个查询、1731个操作数槽中，没有对应值节点位于旧模型查询的两跳入向范围；静态控制也无直接路径。',22,368,'diagnostic_code_v4/semantic_checks.json','下一阶段可设计显式操作数输入和静动态身份关系的独立对照，不能混合改动后直接归因。'),
        ('SMALL_GRAPH_GPU_OVERHEAD','两个小图模型GPU固定批次更慢；原模型逐图搬运、装配与同步成本仍在。',17,283,'gpu_code_v4/server_acceptance.json','后续如确有规模需求，可做不改变信息的批处理工程优化；不是当前研究首要问题。'),
        ('FULL_STATE_AND_INTERVENTION_MISSING','没有完整堆/输入输出状态、通用异常数组库副作用或合法干预前后执行对。',22,368,'outputs/phase05C1_semantic_checks.json','保持T2部分可观测、T3禁用；不能用预测值替代真实观测。')]
    issues=[]
    for code,reason,npd,nt,ev,fix in issue_specs:
        rr=[x for x in raw if x['pre_state_evidence']['opcode']==133] if code=='I2L_NOT_CERTIFIED' else raw if code=='QUERY_VALUE_DISCONNECTED' else []
        if rr:npd=len({tuple(x['audit']['identity'][:3]) for x in rr});nt=len({x['audit']['trace_index'] for x in rr})
        issues.append({'code':code,'reason':reason,'impacted_products':npd,'impacted_traces':nt,'evidence':ev,'root_cause_confirmed':True,'repair_or_next_step':fix,
            'additional_complete_products_recoverable':None,'recovery_reason':'本阶段不修复或重新评估产品级分类准入，不能把局部状态收益换算为新增产品',
            'world_model_impact':True})
    resources={'experiment_commands':commands,'experiment_command_seconds':sum(c['seconds'] for c in commands),
        'peak_worker_rss_bytes':max(c['sampled_peak_process_rss_bytes'] for c in commands),'gpu_allocator':{k:acceptance[k] for k in ('peak_torch_allocated_bytes','peak_torch_reserved_bytes')},
        'research_directory_bytes_at_check':ind['research_directory_bytes'],'environment_allocated_bytes':ind['environment_allocated_bytes'],
        'environment_logical_file_bytes':ind['environment_logical_file_bytes'],'disk_free_bytes':ind['disk_free_bytes'],
        'installation_total_seconds':None,'installation_timing_reason':'未为早期网络重试建立独立总秒表；保留各次日志与文件时间，不能以实验秒数替代安装耗时',
        'dependency_manager_timing_lines':[line for line in (r/'logs/install_mirror_dependencies.log').read_text().splitlines() if line.startswith(('Resolved','Prepared','Installed'))],
        'transfer_archive_bytes':28107214,'uncompressed_export_bytes':298137507,'export_files':8710,'local_export_seconds':59.75,
        'transfer_total_seconds':None,'transfer_timing_reason':'多次SCP及探针未统一计时；主要归档上传和摘要核验解包命令实际墙钟4.634秒，Python包上传2.900秒，非全流程总和',
        'experiment_timeouts':False,'experiment_OOM':False,'truncation':False,'install_network_timeouts':True,
        'hard_resource_quota':False,'monitoring':'0.5秒主进程驻留内存、剩余磁盘和累计7200秒；内部另检查8GiB；研究文件总量事后核验小于4GiB。安装单独计空间。'}
    oldreview=load(r/'snapshot/code/outputs/phase05B_review_packet.json')
    split={'grouped_bank_pilot_v1':{'status':'READY_FOR_BOUNDED_TRAIN_VAL_MECHANISM_DIAGNOSIS','train':s['source_counts']['train'],'val':s['source_counts']['val'],
        'test':{'products':None,'versions':None,'traces':95,'expanded':False,'reason':'95仅引自旧05B公开交接；本轮不展开测试轨迹或重新统计其产品版本'}},
        'leave_BankAccountTP_out':{'status':'NOT_EVALUATED','train':None,'val':None,'test':None,'reason':'本次唯一协议为冻结Bank训练/验证，不新增其他协议数据读取'},
        'leave_Email_out':{'status':'NOT_EVALUATED','train':None,'val':None,'test':None,'reason':'没有Email数据接入或跨系统评价'},
        'FP_TP_class_counts':None,'FP_TP_reason':'本轮全部产品目标禁止读取；未调用分类监督入口',
        'classification_training_conclusion':'NOT_REEVALUATED','world_model_training_conclusion':'PARTIAL'}
    metrics={'products':22,'versions':13,'traces':368,'prefix_events':17518,'branch_targets':523,'arithmetic_pairs':547,'classification_new_products':0}
    packet={'stage_id':'05C1','stage_status':'COMPLETED_WITH_LIMITATIONS','flags':flags,'actual_scope':s['source_counts'],
        'before_metrics':dict(metrics,local_pre_operand_diagnostic_rows=None,reason='05B未提供此独立执行前操作数接口，不能把既有真实数值记成不存在'),
        'after_metrics':dict(metrics,local_pre_operand_diagnostic_rows=1070,primary_branch_certified=418,primary_arithmetic_certified=472,derived_return_branch_rows=105,i2l_abstentions=75,identity_verified_events=17518),
        'recovered_graphs':{'new':0,'historical16':None,'reason':'本轮不修图、不删除节点边、不重做旧16图验收'},
        'admitted_products':{'new':0,'transition_in_scope':22,'historical_classification_cohort':oldreview['admitted_products']['historical_full_classification_cohort'],'classification_reaudited':False},
        'rejection_reasons':{'primary_branch_return_copy':105,'arithmetic_i2l_uncertified':75,'new_graph_rejections':None,'reason':'诊断弃权与图不合格是不同质量维度'},
        'split_readiness':split,'world_model_readiness':{'T1':'PARTIAL:真实有序事件与有限认证依赖结构可用；不等于因果机制已学习','T2':'PARTIAL:418/523分支直接操作数；另105复制推导；547算术输入直接可见，472运算认证','T3':'BLOCKED:无真实合法干预执行对',
            'future_prefix_isolation':True,'static_dynamic_identity_links_written_to_old_graph':False,'structural_learning_benefit_proven':False},
        'tests_executed':{'final_unique_tests_passed':49,'total_test_executions':66,'passed_executions':60,'initial_path_error_executions':6,'path_errors_retested_passed':6,
            'new_java_executions':0,'frozen_real_traces_audited':368,'real_gpu_models_verified':3,'independent_gpu_batch_probes':3},
        'resources':resources,'changed_files':changed,'artifact_paths':dict(paths,primary_report='outputs/phase05C1_review_packet.md',diagnosis='diagnostic_code_v4',original_failed_gpu='gpu_code_v3/server_acceptance.json',
            completed_gpu='gpu_code_v4/server_acceptance.json',command_receipts='command_logs',code_fingerprints='final_checks/code_receipts.json'),
        'unresolved_issues':issues,'recommended_next_stage':{'priority':1,'task':'在同一执行前局部状态格式下，开展Email有限机制数据接入与可观测性验收','reason':'Bank数据已说明精确语义与操作数的重要性，通用接口已明确；下一步优先检验异质机制适用性，避免围绕同一小验证集反复调优','executed':False},
        'training_acceptance':'READY','research_generalization':'PARTIAL','scientific_causal_claims':'NOT_ESTABLISHED'}
    write(out/'phase05C1_review_packet.json',packet)
    covrows=[]
    for part,title in [('train','训练'),('val','验证')]:
        for task,tname in [('branch','条件分支'),('arithmetic','局部算术')]:
            z=s[part][task];c=z['coverage_counts'];covrows.append([title,tname,z['denominator'],c.get('all_known',0),c.get('partial_known',0),c.get('all_unknown',0),c.get('contains_derived_evidence',0),c['checked'],f"{z['coverage']*100:.2f}%",c['consistent'],0])
    timingrows=[]
    names={'seq_gru':'顺序模型（seq_gru）','no_dep':'移除依赖图模型（no_dep）','rgcn':'完整依赖图模型（rgcn）'}
    for x in models:timingrows.append([names[x['model']],f"{statistics.median(x['cpu_fixed_batch_seconds'])*1000:.3f}",f"{statistics.median(x['gpu_fixed_batch_seconds'])*1000:.3f}",f"{x['speedup_median']:.3f}",f"{x['epoch_seconds']:.3f}",x['epoch_updates'],f"{x['peak_allocated_bytes']/2**20:.2f}",f"{x['peak_reserved_bytes']/2**20:.2f}"])
    amb=[]
    for part,title in [('train','训练'),('val','验证')]:
        for key,label in [('structure','原结构输入'),('plus_opcode','加精确操作码'),('plus_trusted_operands','再加主核验可信操作数')]:
            z=s[part]['branch']['ambiguity'][key];amb.append([title,label,z['groups'],z['conflicting_groups'],z['targets_in_conflicting_groups'],f"{z['majority_correct_upper_bound_count']}/{z['denominator']}"])
    text=f'''# CWM-FP 阶段05C-1：服务器验收与局部状态可观测性结果交接

本阶段已经完成真实容器部署、有界GPU验收和冻结轨迹诊断。**工程迁移可用；算法有效性、跨系统泛化和因果机制仍未得到本阶段证明。** 三个旧模型各完成一次完整训练遍历，总计226次优化器更新（213次完整遍历、13次工程检查），未新增正式模型排名。全部523个分支目标和547组算术配对保留，未读取FP/TP产品目标或展开95条封存测试轨迹。

上一阶段05B的结论仍有效：完整依赖图相对移除依赖图有部分改善，仍弱于顺序模型，共同未见分支收益未被支持。本阶段没有通过换用GPU改写这一结论。

## 1. 管理员指南、真实服务器及路径

实际完整读取了[管理员指南正文](https://fpbpf4.yuque.com/pudkpp/gglf8h/mchs7t?singleDoc)。浏览器工具未能启动；使用指南网页公开加载的只读文档接口取得正文和全部命令块。取得回执为本阶段`receipt.json`，本地原始证据在`guide/`。指南正文要求个人容器、个人目录和用户名开头的容器命名。嵌入的另一份“服务器配置”表因空间限额不可读，没有声称已读取；硬件配置以实际检查为准。正文未给出更严的GPU预约、磁盘或运行时限。

真实SSH连接为指定主机的22端口，用户名为用户指定账户。复用已有密钥，严格校验已有主机记录；没有使用或保存此次提供的密码，没有关闭主机校验、扫描其他端口或修改SSH文件权限。已有SSH配置权限不被客户端接受，因此显式指定用户、端口和既有密钥，绕过配置文件解析，而不是绕过主机身份校验。既有主机ED25519指纹：`SHA256:/E9xrH6FpO2AQXRVpzBxHMzQilTRkpaMbuWH1ofssAE`。

宿主机名`2080TI`；指定容器最初停止，确认属于用户并读完指南后启动，未重启正在运行的容器。容器名`wanghaining_ubuntu_ssh`，ID `{acceptance['container_id']}`，镜像`wanghaining_ubuntu_env:20260304`；Ubuntu22.04。既有绑定挂载为宿主机`/sdc/wanghaining`到容器`/workspace`，可读写；既有配置已暴露GPU。本阶段没有安装宿主机软件，没有修改驱动、Docker服务、挂载或共享权限。

{table(['变量','实测路径'],[[k,v] for k,v in paths.items() if k.isupper()])}

代码实际运行目录为`code_v3`及`code_v4`，数据来自`snapshot/data/05A`、`snapshot/data/05B`，各次结果写到新的独立目录。X盘实际映射到用户的`/sdc/wanghaining`，并不是`/codes`。非空探针`path_probe_host.txt`在Windows、宿主机和容器的内容及SHA256一致：`{acceptance['path_probe_sha256']}`。Windows直接写探针返回拒绝且留下空文件，已保留；改用SSH传输至用户拥有的阶段根目录，未修改权限。向容器root创建的环境子目录直接传输也曾拒绝，随后改用用户目录中转。报告写入同一挂载目录的`outputs/`，可经X盘读取；本地项目保存同字节副本。

容器根文件系统已满，但个人挂载目录充足。所有环境、临时文件、下载缓存和结果都位于该挂载目录。容器访问Python发行包和一个普通依赖地址曾超时：Python公开安装包经Windows下载、SHA256核验后传入容器；PyTorch主包仍来自官方CUDA11.8地址，普通依赖使用容器已有可达的清华镜像。没有复制Windows虚拟环境，没有安装额外图框架或完整CUDA开发工具链。

## 2. 完成的实现及验证范围

新增`src/cwm_fp/models/local_state.py`，在原解释器建立调用帧后、当前指令消费栈或生成结果前截获状态。原`trace.py`、前缀生产器、图编码器及05B模型结构均未改动。当前回调已出现的状态会被拒绝。值来源分为此前直接观测、明确操作码常量、有限复制推导和未知；逐事件、逐帧、逐值版本的身份保留在证据侧，`learning_input`仅含操作码、参数个数、类型、来源、可见掩码及主核验允许的值。

新增部署封装、预算监控、诊断、GPU验收和独立复核脚本。`tests/test_prefix_reader.py`仅修正跨平台输入路径与临时目录：使用显式环境变量指向导出输入，临时测试副本放入系统临时目录，不再向冻结05A目录写临时文件。原始失败版本和日志保留。

真实Bank轨迹验证了12/523分支所在产品的返回值复制追踪影响（实际是12个产品、102条轨迹中的105个分支），以及418个直接操作数分支和472组受支持算术的一致性；见后文精确计数。人工单元用例另外覆盖整数溢出、负数除法与余数、移位宽度、空引用、拒绝引用散列/浮点/宽值、重复执行身份、返回复制、操作数次序与未来回调隔离。这些人工用例不算新增真实Java采集。通用JVM解释、数组副作用、完整异常路径、浮点特殊值和一般宽数值转换未实现。本阶段没有运行新的Java测试。

修改/新增文件清单：

{chr(10).join('- `'+f+'`' for f in changed)}

## 3. 迁移完整性与数据规模

导出8710个文件，未压缩298137507字节，归档28107214字节；容器逐文件核验，结束时再次确认8710个文件未变。原05A/05B清单、独立回执和摘要保持不变；Windows绝对来源路径只作审计信息，实际读取采用原有相对路径。没有伪造测试文件、关闭来源检查或同步运行中的源码目录。没有原始SPL、BRFL、分类标签、私钥、浏览器会话或测试轨迹进入包。

训练17产品、9独立版本、283轨迹、13597个前缀；验证5产品、4独立版本、85轨迹、3921个前缀。全部来自BankAccountTP，合计22产品、13版本、368轨迹、17518个投影事件。没有新增产品或图，也没有改变原图节点、边、事件数或正式分类准入。历史28产品分类队列只引用05B交接，未重新读取其目标或验收。

{table(['分区','前缀数','唯一张量缓存','各前缀节点累计','各前缀边累计','静态控制边累计','动态数据边累计'],[[part,cc['prefixes'],cc['unique_cache_files'],cc['sum_nodes_over_prefixes'],cc['sum_edges_over_prefixes'],cc['relation_counts_over_prefixes']['1'],cc['relation_counts_over_prefixes']['2']] for part,cc in ind['cache_checks'].items()])}

上表是重复历史前缀快照的累计量，不能当作去重后的原始图节点/边数。没有事件、测试或依赖边截断。产品/图状态表分别列出22产品和368轨迹；原始完整套件资格为继承信息，本阶段未重采或独立重验原套件登记，因此相应未知值没有用0代替。

## 4. 真实GPU验收与失败证据

实际Python3.11.13、PyTorch2.6.0+cu118、NumPy2.4.6；驱动535.104.05。使用宿主机序号1的RTX3090Ti，UUID `{acceptance['gpu_uuid']}`；通过可见设备限制，在PyTorch中对应`cuda:0`。容器中能列出显卡不是验收依据：已记录真实模型进程、设备UUID、有效梯度、参数变化、GPU显存及输出目录。未停止其他用户任务或使用多卡。

复用了旧`phase05B_linux_acceptance.py`的实际命令帮助及一次真实更新，再完成三个模型相同初始参数、相同固定187前缀/4轨迹批次的CPU/GPU对照。图编码器、顺序循环网络及两预测头均有有效梯度和参数更新；旧检查点及其训练区保存输出核对通过，旧优化器计数恢复后正确递增。真实4轨迹独立/合批/逆序对照通过，未发现跨轨迹动态连接。

数值容差在首次试跑前声明：输出绝对/相对2e-5，损失绝对2e-6/相对2e-5，梯度3e-5，更新参数1e-4。首次图模型恢复检查另设的零容差逐位输出检查失败，最大差1.9073486328125e-6；失败留在`gpu_code_v3/`。继续核查发现同一实例重复前向也有同量级差异，恢复后的状态张量逐位一致。后续保留逐位输出失败状态，用原先已声明的输出容差判断功能恢复，没有提高任何原定数值容差。该行为与图汇聚使用CUDA并行索引求和的非逐位确定性一致；[PyTorch官方说明](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html)也提示这一限制。不能把“在容差内恢复”写成图模型输出逐位复现。

已完成的顺序模型整轮未重跑；只继续两个未完成的图模型。三者各且仅一轮完整遍历，种子20261101，每轮71次更新，完整覆盖283轨迹和13597前缀。总计226次更新，其中包含失败尝试和CPU对照。工程输出均标记`server_acceptance_only`，未使用验证目标选择检查点、未生成新泛化排名。

{table(['模型','固定批次CPU中位毫秒','GPU中位毫秒','CPU/GPU比值','完整一轮秒','整轮更新','峰值已分配MiB','峰值保留MiB'],timingrows)}

固定批次先预热，再同步计时3次，包含旧读取接口的逐图装配。比值小于1表示GPU更慢；两个小图模型没有加速。GPU设备算子时长由一次额外、无更新的性能记录得到：顺序约4.443毫秒、移除依赖图12.328毫秒、完整图15.693毫秒；这与端到端时长不同。主机装配、逐图搬运以及原代码为了静态结构指纹进行的同步仍占时间，不能把同步墙钟全称为GPU计算。单独数组准备/传输探针约3.892/8.275毫秒；缓存读取约6.066秒。这些分别测量的探针不是可直接相加的全流程分解。

## 5. 执行前局部状态与精确语义

规则和字段在训练材料及已有解释器基础上固定，训练处理完成后对验证应用同一规则。完整保留430训练+93验证分支、402训练+145验证算术配对。输入先由真实物理前缀产生，目标在该轨迹输入截获完成后才连接。比较按栈底到栈顶的value1/value2次序；整数按32位有符号规则处理，溢出截断、除法向零取整。不支持操作明确弃权，未用分支结果反推输入，也未使用当前指令未来回调。规则依据[Java虚拟机指令规范](https://docs.oracle.com/javase/specs/jvms/se8/html/jvms-6.html)，实际是否成立由本轮轨迹核验。

{table(['分区','任务','完整分母','主核验全部已知','部分已知','主核验全部缺失','含复制推导','实际核验','核验覆盖率','一致','不一致'],covrows)}

“主核验全部缺失”只指直接观测/静态常量这一严格层级，不等于没有任何证据：105个分支的输入来自整数返回指令`ireturn`生成的复制值版本，之前标量来源可追溯。89个训练分支影响10产品、6版本、86轨迹；16个验证分支影响2产品、2版本、16轨迹。已独立重放全部105条并核对来源、帧和可见时刻，将其标记为`SEMANTICALLY_DERIVED`，没有冒充新的直接测量；仍从主符号核验弃权。真实未知值类型没有补零。

主分支核验合计418/523，覆盖79.92%，覆盖内418/418一致；不能写成523个分支全部认证。分支操作数共610个直接观测槽、105个返回复制推导槽。算术547组输入全部有直接观测，共1016个槽，其中472组受支持操作一致；其余75组为`i2l`（训练51、验证24），整数转长整数未纳入本轮认证，不是输入缺失。直接观测的算术槽中89个也具有操作码内嵌常量证据（训练61、验证28），属于冗余来源；静态常量额外增加的完整主核验覆盖为0。静态常量没有被计作运行测量。

初版诊断漏识别解释器既有`return_value`角色，后修正复制证据分类及其完成可见时刻，并新增专项测试。初版和修正版都保留，418和472这两个主核验数没有变化。未将推导值加入本阶段旧模型输入。

各精确操作码、两种分支结果和全部目标证据可在`phase05C1_semantic_checks.json`及1070行状态CSV逐项核查。没有任何符号核验不一致需要通过改目标或补输入修正。

## 6. 表示歧义与静动态连接

{table(['分区','输入','不同输入组','冲突组','冲突组内目标数','经验多数上界计数'],amb)}

精确操作码只带来有限变化；组拆分后，冲突组数可以增加，不能只用组数升降判断改善。加入主核验可信操作数后，本队列精确输入组的目标冲突消失；保留的未知掩码未用唯一身份区分。**这是有限样本的表示诊断，不是神经模型成绩、总体可预测性证明或缺陷检测收益。** 105个返回复制位置仍不参加主语义核验，其相同有限输入未冲突不能替代语义证明。

全部17518动态事件都能以真实类、方法描述符、字节码位置及冻结类文件摘要连接唯一规范静态指令；源行不被当作语句级认证。12534个事件对应产品静态输入节点（训练9770、验证2764）；其余4984个对应测试代码的静态身份目录，未伪造为产品节点。没有新增边到旧图。

1070个诊断查询中，987个可经独立身份侧表定位产品静态上下文。旧两层网络查询的入向两跳范围内，1731个操作数槽对应的值节点为0，定义事件1676个；1015个查询的全部定义事件在两跳内。静态控制节点直接可达数为0。旧模型仍有静态全图汇聚和动态事件汇聚，因此不能据此称其完全不使用静态信息；准确限制是缺少查询与具体静态指令、执行前值节点之间的局部关系。侧表`event_of_static_instruction.jsonl`不是动态控制依赖，更不是因果图认证。

## 7. 测试、资源与复现

最终49项不同测试通过：新增局部状态12项、原前缀观察20项、前缀读取6项、转换模型11项。过程共66项执行，60次通过，6次因旧测试硬编码Windows路径失败，修复后6/6通过；失败未覆盖。旧Linux入口帮助、真实单步、两次诊断、有限训练和独立检查均有单独日志及命令回执。真实Java新执行数为0；复用368条冻结真实轨迹。独立检查包含8710文件不变、所有训练/验证张量缓存读取、1070条时间/字段检查、105条复制重放和3种模型的真实GPU批次隔离。

已记录实验命令累计{resources['experiment_command_seconds']:.3f}秒，不含人工编码、环境安装和网络传输。最大实测工作进程驻留内存{resources['peak_worker_rss_bytes']/2**20:.2f}MiB；PyTorch峰值已分配{acceptance['peak_torch_allocated_bytes']/2**20:.2f}MiB、保留{acceptance['peak_torch_reserved_bytes']/2**20:.2f}MiB，不包含完整CUDA上下文的全部设备占用。研究目录核验时{ind['research_directory_bytes']}字节（{ind['research_directory_bytes']/2**30:.3f}GiB）；环境实际分配约{ind['environment_allocated_bytes']/2**30:.3f}GiB，逻辑文件累计{ind['environment_logical_file_bytes']/2**30:.3f}GiB（缓存与环境存在硬链接，不能重复当作物理占用）；剩余磁盘{ind['disk_free_bytes']/2**30:.1f}GiB。依赖环境明确不计入4GiB研究数据预算。

安装整体和所有传输没有统一秒表，机器报告将总耗时记为null并说明原因，保留安装管理器的实际计时行及失败日志；未把实验秒数冒充部署总时长。没有实验超时、日志截断、OOM或缓存失效；发生过安装网络超时。资源限制通过0.5秒主进程监测、内部检查和最终目录核验实现，未施加系统硬配额，也未独立测量安装进程峰值或所有短暂子进程内存。

实际完整命令与退出码在机器报告`resources.experiment_commands`和`command_logs/*.json`。容器入口为`bash phase05C1_run_v3.sh <动作> <冻结代码目录>`，日志列出展开后的Python绝对路径和参数。已经存在的结果目录会拒绝覆盖；复现须另建阶段目录并核验代码、导出回执及新输出路径。环境锁定在`final_checks/requirements-linux.lock`，旧快照与新代码摘要在`final_checks/code_receipts.json`。

## 8. 研究准入与下一步唯一建议

**服务器工程验收：可用（READY）。研究泛化与完整世界模型数据：部分具备（PARTIAL）。** 分组协议下17训练产品/9版本、5验证产品/4版本可用于已限定的机制诊断；测试区保持封存，没有新测试区成绩。留出BankAccountTP与留出Email两协议本阶段未执行，训练/验证/测试合格数为未测量，不能填0冒充统计。FP/TP类别覆盖未读、未重评，不能从本阶段声明正式下游分类已具备条件。

- 结构事件层面（T1）：真实有序事件、有限认证依赖及时间前缀可用，身份连接可靠；尚无证据表明神经模型学到因果机制。
- 数值状态层面（T2）：已有明确的局部操作数和部分算术观测，可设计受限状态任务；不是完整当前堆、完整输入输出断言或一般状态转换，105个分支依赖复制推导、75组宽转换未认证。
- 合法干预层面（T3）：没有真实合法干预前后执行对，继续禁用干预训练和反事实有效性结论。没有Email训练、跨系统评价、FP/TP联合训练或强化学习。

操作数齐备时，分支已有可执行语义参照；学会这些规则不等于发现缺陷。覆盖内一致不证明跨系统泛化，缺少直接观测也不允许用预测值填充“真值”。未来状态任务必须继续用当前边界内已到达证据，不能提前输入最终输出、后续依赖或未来事件。

**唯一首要建议：在本阶段已明确的执行前局部状态格式下，开展Email有限机制数据接入与可观测性验收。** Bank数据已经表明粗操作类别与缺少显式操作数会限制任务解释；继续围绕同一个小验证集追加架构试验收益有限。先检查这种字段、类型、时刻和缺失规则能否在异质代码上成立，再冻结精确语义与相同信息的局部状态预测对照。前提是下一阶段单独授权，并对Email已有数组、异常和库副作用缺口保持拒绝，不能据此宣布Email已可训练。当前可以准备受限GPU实验，但不建议因部署成功就追加05B训练；两个小图模型的GPU效率也仍需按实际规模判断。

所有未解决事项、影响范围、证据和可修复性在机器报告`unresolved_issues`逐项列出。本阶段没有新增任何正式分类准入产品；没有把局部状态认证换算成产品恢复收益。原数据、冻结划分、旧正式结果、初版诊断与失败日志均保留；没有删除历史文档。到此停止，未启动下一阶段。
'''
    # Correct the narrative to keep products and branch denominators distinct.
    text=text.replace('真实Bank轨迹验证了12/523分支所在产品的返回值复制追踪影响（实际是12个产品、102条轨迹中的105个分支）','真实Bank轨迹验证了12个产品、102条轨迹中105个分支的返回值复制追踪')
    (out/'phase05C1_review_packet.md').write_text(text,encoding='utf8')
    for p in out.glob('*.json'):load(p)
    hashes={p.name:digest(p) for p in out.iterdir() if p.is_file()};write(r/'report_hashes.json',hashes)
    print(json.dumps({'reports':len(hashes),'status':packet['stage_status'],'experiment_seconds':resources['experiment_command_seconds']}),flush=True)
if __name__=='__main__':main()
