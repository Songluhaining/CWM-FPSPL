"""Verify matched holdout outputs and write one Chinese stage handover."""
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
import numpy as np
from cwm_fp.data.common import ROOT,KEY,sha256,write_csv,write_json
from cwm_fp.models.matched_evaluation import read,verify_protocol,verify_predictions


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    protocol=ROOT/'outputs/phaseEVAL1_protocol_v2.json'
    p=verify_protocol(protocol);verify_predictions(p);old=read(p['old_protocol']);work=Path(p['out'])
    results=ROOT/'outputs/phaseEVAL1_results';receipt=read(results/'evaluation_receipt.json')
    for name,digest in receipt['files'].items():
        if sha256(results/name)!=digest:raise ValueError('Result changed: '+name)
    preds=csv_rows(results/'product_predictions.csv');metrics=csv_rows(results/'metrics_by_seed.csv')
    groups=defaultdict(list)
    for r in preds:groups[r['model'],r['seed'],r['partition']].append(r)
    for r in metrics:
        rows=groups[r['model'],r['seed'],r['partition']];counts=Counter((int(x['y_true']),int(x['y_pred'])) for x in rows)
        tn,fp,fn,tp=[counts[x] for x in ((0,0),(0,1),(1,0),(1,1))]
        assert len(rows)==int(r['n']) and tp+fn==int(r['FP']) and tn+fp==int(r['TP'])
        assert [tn,fp,fn,tp]==[int(r[k]) for k in ('tn','fp','fn','tp')]
        assert math.isclose(2*tp/max(1,2*tp+fp+fn),float(r['fp_f1']),abs_tol=1e-12)
    package=Path(old['paths']['package']);manifest=read(package/'manifest.json')
    graph_status=csv_rows(ROOT/'outputs/phaseEVAL1_graph_status.csv')
    status_index={tuple(r[k] for k in (*KEY,'test_id')):r for r in graph_status}
    totals=defaultdict(Counter);versions=defaultdict(set);product_rows=[]
    for prod in manifest['products']:
        identity=prod['identity'];part=prod['partitions']['grouped'];counts=Counter()
        assert len(prod['tests'])==len(prod['registered_tests'])
        for test in prod['tests']:
            tpath=package/test['tensor']['path'];assert sha256(tpath)==test['tensor']['sha256']
            g=status_index[tuple(test['identity'])]
            with np.load(tpath,allow_pickle=False) as a:
                assert a['x'].shape==(test['nodes'],22) and a['edge_index'].shape==(2,test['edges'])
                observed={name:int(np.sum(a['edge_type']==i)) for i,name in enumerate(manifest['relations'])}
                assert all(observed[name]==test['relations'].get(name,0) for name in observed)
            assert all(int(g[k])==test[k] for k in ('nodes','edges','events'))
            counts.update(graphs=1,nodes=test['nodes'],edges=test['edges'],events=test['events'],
                control_edges=observed['control_dep'],data_edges=observed['data_dep'])
        versions[part].add(tuple(identity[:2]));totals[part].update(counts);totals[part]['products']+=1
        product_rows.append(dict(zip(KEY,identity))|{'sample_id':prod['sample_id'],'partition':part,
            'registered_tests':len(prod['registered_tests']),'read_tests':counts['graphs'],'complete_suite':True,
            'graph_identity_valid':True,'supported_dependency_semantics_valid':True,'lossless':True,
            'prediction_available':True,'reason':'','evidence_file':str(package/'manifest.json')})
    assert len(product_rows)==28 and len(status_index)==463
    for part in totals:totals[part]['versions']=len(versions[part])
    write_csv(ROOT/'outputs/phaseEVAL1_product_status.csv',product_rows,product_rows[0].keys())
    graph=read(work/'graph_prediction_receipt.json');pu=read(work/'pu_prediction_receipt.json')
    commands=read(ROOT/'outputs/phaseEVAL1_v2_commands.json');assert commands['completed']
    graph_methods={'rgcn':'现有关系图分类器','untyped':'合并关系图对照','nodeonly':'无边节点对照',
        'clap6_svm':'CLAP六维支持向量适配','pulp50_lr':'PULP五十维逻辑回归','pulp50_svm':'PULP五十维支持向量',
        'coverage2_lr':'测试数与覆盖率对照','Stats-LR':'图统计逻辑回归','Prior':'训练比例常数',
        'pulp_pu_extra_train_unlabeled':'PULP源码（额外训练区无标签池）'}
    means=receipt['metrics'];by={(r['model'],r['partition']):r for r in means}
    order=['rgcn','clap6_svm','pulp50_svm','pulp50_lr','coverage2_lr','untyped','nodeonly','Stats-LR','Prior','pulp_pu_extra_train_unlabeled']
    def fmt(value):return '不适用' if value is None else f'{value:.4f}'
    rows=[]
    for model in order:
        r=by[model,'test']
        rows.append([graph_methods[model],r['seeds'],fmt(r['fp_precision_mean']),fmt(r['fp_recall_mean']),fmt(r['fp_f1_mean']),
            fmt(r['accuracy_mean']),fmt(r['average_precision_mean']),fmt(r['log_loss_mean'])])
    score_table=table(['方法','种子数','精确率','召回率','F1','准确率','平均精确率','概率对数损失'],rows)
    counts_table=table(['分区','版本','产品','FP','TP','测试图','节点','边','事件','控制边','动态数据边'],
        [[part,totals[part]['versions'],totals[part]['products'],receipt['truth_counts'][part].get('FP',0),receipt['truth_counts'][part].get('TP',0)]+
         [totals[part][k] for k in ('graphs','nodes','edges','events','control_edges','data_edges')] for part in ('train','val','test')])
    cmd_table=table(['实际步骤','退出码','秒数'],[[r['name'],r['exit_code'],f"{r['seconds']:.3f}"] for r in commands['commands']])
    size=sum(f.stat().st_size for folder in (work,results,ROOT/'data/processed/labels/eval1_matched_v1') for f in folder.rglob('*') if f.is_file())
    resources={'new_data_and_results_bytes':size,'free_bytes':{drive:shutil.disk_usage(drive+':/').free for drive in ('D','E')},
        'graph_worker_memory':graph['process_memory'],'graph_inference_seconds':graph['seconds'],
        'successful_command_wall_seconds':commands['wall_seconds'],'total_stage_wall_seconds':None,
        'null_reason':'No common timer over audit, code changes, report authoring and verification',
        'GPU_used':False,'server_changes':False,'graph_optimizer_updates':0,'new_java_test_executions':0,
        'timeouts':False,'OOM_observed':False,'truncation':False}
    validation={'passed':True,'prediction_rows':len(preds),'model_seed_runs':26,'metric_rows_recomputed':len(metrics),
        'products_per_run':28,'graphs_reverified':463,'graph_counts':{k:dict(v) for k,v in totals.items()},
        'selected_graph_restore_max_difference':max(c['validation_max_abs_difference'] for c in graph['checks']),
        'graph_restore_tolerance':p['graph_atol'],'source_and_predictions_still_match_frozen_hashes':True}
    write_json(ROOT/'outputs/phaseEVAL1_validation.json',validation)
    changed=['src/cwm_fp/models/matched_evaluation.py','scripts/run_matched_evaluation.py',
        'scripts/report_matched_evaluation.py','tests/test_matched_evaluation.py','DATA_CONTRACTS.md','EVALUATION_PROTOCOL.md','STAGES.md','outputs/phase_status.md']
    packet={'stage_id':'EVAL1','stage_status':'COMPLETE_WITH_LIMITED_SCOPE','actual_scope':'Frozen BankAccountTP graph classifier holdout and matched-cohort baselines; not unified world-model training',
        'before_metrics':{'graph_test_predictions':0,'products':28,'graphs':463,'historical_validation_products':5},
        'after_metrics':{'products':28,'graphs':463,'new_products':0,'new_graphs':0,'test_products':6,'test_versions':3,
            'model_seed_runs':26,'predictions':728,'graph_optimizer_updates':0,'current_rgcn_test_f1':by['rgcn','test']['fp_f1_mean'],
            'clap_test_f1':by['clap6_svm','test']['fp_f1_mean'],'pulp50_svm_test_f1':by['pulp50_svm','test']['fp_f1_mean']},
        'recovered_graphs':0,'admitted_products':{'existing':28,'new':0},'rejection_reasons':[],
        'split_readiness':{'grouped':{'status':'PARTIAL','reason':'Only a small single-system pilot','counts':{k:dict(v) for k,v in totals.items()},'labels':receipt['truth_counts']},
            'loso_BankAccountTP':{'status':'BLOCKED','train_products':0,'validation_products':0,'test_products':28},
            'loso_Email':{'status':'BLOCKED','train_products':25,'validation_products':3,'test_products':0},
            'four_system_protocol':'NOT_EVALUATED_FOR_GRAPH_MODEL'},
        'world_model_readiness':{'unified_design_status':'DESIGNED_NOT_IMPLEMENTED','T1':'Existing admitted structures only, unchanged',
            'T2':'Local observations exist historically; joint state/classification model not trained','T3':'No verified intervention pairs',
            'causal_effect_validated':False,'cross_system_model_evaluated':False,
            'temporal_prefix_warning':'Whole-test classification inputs cannot be reused as early transition prefixes'},
        'tests_executed':{'new_boundary_tests_passed':7,'existing_graph_comparison_regressions_passed':12,
            'total_distinct_passed':19,'real_checkpoint_restores':9,'frozen_statistical_models_restored':2,
            'original_java_tests':0,'logs':['outputs/phaseEVAL1_unit_tests_v2.log','outputs/phaseEVAL1_graph_regression.log']},
        'resources':resources,'commands':commands['commands'],'validation':validation,'changed_files':changed,
        'artifact_paths':{'report':'outputs/phaseEVAL1_review_packet.md','csv':'outputs/phaseEVAL1_results/当前模型与基线对比.csv',
            'predictions':'outputs/phaseEVAL1_results/product_predictions.csv','products':'outputs/phaseEVAL1_product_status.csv',
            'graphs':'outputs/phaseEVAL1_graph_status.csv','protocol':'outputs/phaseEVAL1_protocol_v2.json'},
        'unresolved_issues':[
            {'issue':'Only 6 test products / 3 versions / 1 FP','impact':'No statistically strong conclusion or stable cross-system claim'},
            {'issue':'Unified shared-encoder transition/classification design not implemented','impact':'Current result tests graph classification only'},
            {'issue':'Baseline observable failure context and extra PU unlabeled pool','impact':'Same labeled cohort is not identical information availability'},
            {'issue':'Graph admission covers only BankAccountTP subset','impact':'Cannot compare directly to full four-system BASE4 scores'},
            {'issue':'Prior Bank/Email development and BASE4 exposure','impact':'Cannot describe entire research test population as never examined'}],
        'recommended_next_stage':'Implement the frozen minimal shared classifier/transition design and classification-only control on a fixed matched cohort; expand independently admitted system data before strong claims',
        'test_tuning_performed':False,'performance_claim':'Current R-GCN did not outperform CLAP/PULP50-SVM on this frozen pilot'}
    write_json(ROOT/'outputs/phaseEVAL1_review_packet.json',packet)
    report=f'''# 当前模型与基线同队列验证（EVAL1）

日期：2026-11-02。已完成实际验证。结论是：**现有关系图分类器可以运行并具有识别信号，但在本次6个测试产品上没有超过CLAP和PULP五十维支持向量对照。最新统一世界模型仍是已设计、未实现的联合模型，不能把这次成绩称为它的性能。**

## 1. 当前究竟有什么模型，为什么现在可以先验证

当前已实现并训练过的是两层、隐藏宽16的关系图分类器（R-GCN），读取22列语义/布尔节点特征与7类关系，先按测试内节点均值汇聚，再对完整产品的全部测试等权汇聚，输出FP概率。它没有使用当前设计中的精确操作码、32位操作数表示、静态到动态身份传递和联合局部分支/数值损失。

阶段04R1曾在17训练产品上拟合、用5验证产品选择检查点；关系图模型验证F1为1，但没有评价其6个独立测试产品，也没有在同队列与CLAP/PULP比较。阶段05B另行训练过局部转换预测器，尚未与产品分类联合；其转换效果不能直接当作FP识别效果。

本轮不等待模型设计“完全完善”，先对可运行部分完成缺失的同队列验证。复用全部9份已选神经检查点和2个已冻结统计模型，没有重新训练神经网络、调阈值或换检查点。最新设计足够明确，可以进入最小联合实现与消融；当前缺的是实现与可用数据覆盖，而不是再添加未经验证的理论结构。

## 2. 实际范围与数据隔离

仅复用原04D1/04R1冻结的BankAccountTP合格包，28产品、16独立版本、463测试图。没有采集其他版本，没有删测试、修改标签、重划分或覆盖旧结果。每个产品完整套件与既有包外摘要核验，新增合格产品0、恢复图0。

{counts_table}

新四系统基线的178版本/4206产品结果保持冻结。本轮是其中特定旧队列上的图模型对照，**不是四系统全量图实验**。Email、Elevator、ExamDB没有在本轮成为图模型的训练或测试系统。

训练区只使用17个通过产品的FP/TP目标；验证区5个产品选择表格模型参数；图模型沿用旧选择，不再选。全部26组模型与种子组合的预测先封存，再打开此次测试标签导出并评价。真实缺陷位置、版本/产品身份及标签来源没有进入模型特征。身份仅用于连接和审计。

6个测试产品来自与训练/验证不重叠的3个版本，但BankAccountTP已长期用于开发，且BASE4曾在另一协议下评价重叠产品。因此只能说本轮冻结图检查点首次接受此测试评价，不能称整个研究从未接触这批数据或宣称全新外部验证。

## 3. 公平比较做到什么，哪些信息条件仍不同

CLAP、PULP五十维逻辑回归/支持向量、测试数/覆盖率对照，都只用同17个通过产品的监督目标，并在同5验证、同6测试产品上评价。三个种子沿用图试验20261101/02/03。CLAP线性支持向量C=1；其他三种表格对照仅在冻结的C=0.1、1、10中按验证F1选择，平分取较小C。神经检查点沿用04R1验证概率损失选择，选择准则不同但均不看测试结果。

表格特征逐元素复用BASE4已封存的源码规则特征，不重新根据测试成绩改造。CLAP/PULP特征允许同缺陷版本的原始失败测试上下文；训练外层缩放使用17个通过产品及同9个训练版本的91个原始失败参考。图模型本身没有额外输入这些失败产品。因此这里匹配的是监督产品与评价分母，不是完全相同信息量；不能将差异简单归结为网络结构。

原PULP的5%暂定负例抽样对17个无标签通过产品取整为0，严格17池无法运行。本轮预先单列一个扩展无标签条件：仅同9个训练版本内的91个F、215个原始通过产品，共306个可观察产品，运行原30分类器、5%抽样算法；没有读取这些额外产品的FP/TP来拟合，没有用验证/测试版本扩充池。该项明确标记“额外训练区无标签池”，不冒充与其他方法完全同训练输入。PULP五十维监督模型是对照，不是PULP原算法。

## 4. 独立测试结果

测试区真实标签为1个FP、5个TP，固定阈值不变。下表为全部预定种子的均值，统计模型只运行一次；支持向量距离不是校准概率，概率损失留空。

{score_table}

关系图分类器三个种子均得到混淆矩阵 `[[4,1],[0,1]]`（行真实TP/FP，列预测TP/FP）：唯一FP被识别，但5个TP中误报1个。精确率0.5、召回率1、F1为0.6667、准确率0.8333。CLAP和PULP五十维支持向量对照在这6个产品上均无错误，F1为1。因此目前不能声称图模型超过最强传统基线。

关系图模型相对无边节点对照的F1为0.6667对0.5，相对合并关系图的F1相同；概率损失分别约0.4856、0.7120、0.5278，当前关系图模型概率质量更好。其与关系合并模型还存在按关系归一化差异，不能把全部收益归因于关系标签，更不能解释为因果机制得到识别。

只有一个正样本会使排序面积容易达到1，F1也会因一次误报大幅改变。保留按版本重采样200次的区间及有效重复数，但仅3个版本簇，不能据此作强显著性结论。三个种子不是18个独立测试产品。不会因为这次误报再调整阈值或重新挑选模型。

结果CSV：[当前模型与基线对比](phaseEVAL1_results/当前模型与基线对比.csv)。训练、验证、测试共30行汇总；另有78行逐种子指标、728行逐产品预测以及26组版本簇区间。验证结果用于选择过模型，不能代替本节测试结果。旧四系统总体成绩不能直接与本次6产品结果相减。

## 5. 实际改动、检查和失败记录

新增 `models/matched_evaluation.py`，作为独立授权的模型恢复、同队列特征连接、分区标签导出和测试评价入口。它复用原图读取器、分类器、检查点加载器、CLAP/PULP实现、标签接口和指标；没有修改图网络结构、动态语义或旧阶段禁止测试读取的入口。新增有界调度与报告脚本、7项身份/分区/摘要边界测试，更新契约及阶段状态。

实际检查：9份神经检查点恢复后的5验证预测均在预先固定1e-6绝对容差内复现，最大差{validation['selected_graph_restore_max_difference']:.10g}；R-GCN三次差均0。参数逐张量检查不变，更新次数为0。常数与统计逻辑回归亦从旧参数恢复并核对验证输出。463图全部无损读取，重新核对张量摘要、22列宽度、边数及各关系数量；完整测试没有截断。728条预测每组28产品齐全，78行混淆矩阵和F1独立复算一致。

新增测试7项、既有图比较回归12项，共19项不同测试通过。最初冻结后，分区导出成功写出训练标签，但自己的摘要读取被标签隔离规则拒绝，首轮退出1。修复仅允许当前导出的同一分区回读校验，其他分区仍禁止；保留旧代码、失败日志、旧协议，新协议v2引用父摘要。修复前没有图推理、基线拟合或测试标签读取；没有依据测试结果改代码或实验方案。第一次6项测试及修复后7项分别保留，19为最终不同测试数量。

## 6. 命令、环境和资源

入口：`python -B scripts/run_matched_evaluation.py --statistics-python D:/SoftInstall/python.exe --graph-python E:/Myself/投稿/work7/FP-SPL/.venv/Scripts/python.exe`。成功续跑增加 `--protocol outputs/phaseEVAL1_protocol_v2.json --run-id phaseEVAL1_v2 --resume-after freeze`。每步完整参数、退出码、日志路径在 `phaseEVAL1_v2_commands.json`。旧失败记录为 `phaseEVAL1_commands.json`，不覆盖。

{cmd_table}

实际Windows CPU执行。图恢复推理用PyTorch2.6.0+cu126、NumPy2.4.6；表格用scikit-learn1.3.0、NumPy1.24.3，均Python3.11.4。使用已安装环境，没有安装依赖或操作服务器。单计算线程；单命令超时上限600秒，研究数据预算256MiB，未施加操作系统硬内存配额。

成功命令流水线墙钟{commands['wall_seconds']:.3f}秒；图恢复/推理内部计时{graph['seconds']:.3f}秒，图工作进程生命周期峰值常驻内存约{graph['process_memory']['peak_working_set_bytes']/1024**3:.3f}GiB。没有GPU计算或新的神经训练，本轮无需等待服务器训练环境。表格基线新拟合30个候选及3次PULP模型组，旧神经参数不更新。整个审计、开发和报告阶段没有统一起止计时，记空值，不把推理秒数冒称全部耗时。

新数据及结果{size:,}字节，核验时E盘余量{resources['free_bytes']['E']/1024**3:.3f}GiB、D盘{resources['free_bytes']['D']/1024**3:.3f}GiB。正式成功执行未见超时、内存溢出、NaN或静默丢样本。没有重跑Java。所有原始包、检查点、标签和旧实验结果保持冻结。

## 7. 距离完整研究目标还差什么

现在可以确认“当前图分类器的实际独立测试表现”，不能确认“统一因果世界模型有效”。设计中的共享编码器＋产品分类＋可信分支/整数辅助头尚未形成可运行联合实现，最新配置仍标为 `DESIGNED_NOT_IMPLEMENTED`。05B的旧转换预测实验不能替代这项联合学习检验。

完整四系统图覆盖尚不具备：现有合格包只有BankAccountTP的28产品；留出BankAccountTP时没有图训练/验证产品，留出Email时没有图测试产品。Elevator/ExamDB的传统基线输入已齐，不等于真实LDDG和状态已齐。图模型的完整套件认证、Email状态/前缀缺口及跨系统共同样本覆盖，均不能靠这次评价解决。

T1：既有有序事件和依赖图可用于这批分类产品，未新增。T2：历史局部数值诊断部分可用，当前图分类22列没有完整数值状态，未验证联合状态预测收益。T3：本轮无真实合法干预对。分类在原始测试全部完成后使用完整图是允许的；这些最终图、测试结果和后验统计不能直接成为较早转换时间步输入。

下一步优先落实已经冻结的最小联合模型，并保留输入、产品、训练预算完全相同的“只分类”对照，检验辅助转换损失是否真正改善产品识别。第二优先完成其他系统的最小可靠图/状态接入，保持全量拒绝统计与共同子集基线；没有共同队列就不宣称四系统公平比较。第三是按版本扩大独立测试支持数；当前测试已揭盲，后续设计不能围绕这一个误报反复调优。无需继续空泛扩充架构，也没有依据现在直接开展反事实或强化学习。

## 8. 交付与停止边界

完整交接为本文及同名JSON；另有产品状态28行、图状态463行、冻结协议v1/v2、命令记录、独立核验及结果CSV。`phaseEVAL1_summary.md`只作索引。未发现可安全删除且无独立用途的旧文档，失败与冻结历史保留。源码按既有版本控制要求提交，结果和标签留在本地。

本轮验证完成，不自动实现联合模型、扩大采集或追加训练。
'''
    (ROOT/'outputs/phaseEVAL1_review_packet.md').write_text(report,encoding='utf-8')
    (ROOT/'outputs/phaseEVAL1_summary.md').write_text('# 当前模型验证总结\n\n已实际完成现有图分类器与传统基线在同一BankAccountTP队列上的比较。6个测试产品中，图模型F1为0.6667，CLAP与PULP五十维支持向量对照均为1.0000。样本很小，不能作跨系统结论。\n\n[完整中文交接](phaseEVAL1_review_packet.md)；[结果CSV](phaseEVAL1_results/当前模型与基线对比.csv)。最新统一世界模型尚未联合实现，本轮没有重新训练神经模型。\n',encoding='utf-8')
    print(json.dumps({'verified':True,'products':28,'graphs':463,'predictions':len(preds),'metric_rows':len(metrics),'report':'outputs/phaseEVAL1_review_packet.md'}))


if __name__=='__main__':main()
