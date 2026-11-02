"""Bounded Java8 semantic acceptance fixtures; all products here are synthetic."""
from __future__ import annotations
import json
import argparse
import os
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cwm_fp.data.common import write_json
from cwm_fp.lddg.runner import checked,execute_pair,command,completion
from cwm_fp.lddg.trace import parse_trace,Dataflow
from cwm_fp.lddg.static import load_static


def main():
    c=json.loads((ROOT/'configs/lddg_smoke.json').read_text(encoding='utf-8'))
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'data/processed/lddg/phase03_synthetic_v2')
    dest=parser.parse_args().out.resolve()
    if not dest.is_relative_to(ROOT/'data/processed/lddg'):raise ValueError('Fixture output outside generated root')
    work=ROOT/c['work_dir'];dest.mkdir(parents=True,exist_ok=True)
    variant=dest/'variant'
    for folder,source in [('src','Phase03Program.java'),('test','Phase03FixtureTest.java')]:
        (variant/folder).mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'tests/java'/source,variant/folder/source)
    java=Path(c['java_home'])/'bin';producer=ROOT/'third_party/existing_instrumenter';classes=work/'producer-classes';agent=work/'agent.jar';build=dest/'build'
    for part,folder in [('main','src'),('tests','test')]:
        (build/part).mkdir(parents=True,exist_ok=True)
        cp=os.pathsep.join(map(str,[build/'main',producer/'lib/*']))
        checked([java/'javac.exe','-g','-encoding','UTF-8','-cp',cp,'-d',build/part,*sorted((variant/folder).glob('*.java'))],dest,dest/(part+'-build.log'))
    cp=os.pathsep.join(map(str,[classes,producer/'lib/*']))
    checked([java/'java.exe','-cp',cp,'Phase03StaticExport',dest/'static.tsv',*sorted(build.rglob('*.class'))],dest,dest/'static-export.log')
    static=load_static(dest/'static.tsv');results=[]
    methods=['assignmentChain','trueBranch','falseBranch','loopWrites','recursiveCalls','heapAlias','assertionFailure','unsupportedArray','missingStringValue','optimizedLine']
    for name in methods:
        identity='phase03fixture.Phase03FixtureTest.'+name
        record=execute_pair(c,work,producer,classes,agent,variant,identity,['phase03fixture.Phase03Program'],build,dest/name,'FAIL' if name=='assertionFailure' else 'PASS')
        if not record['parity']:raise AssertionError('Instrumentation changed fixture outcome: '+name)
        raw=parse_trace(dest/name/'instrumented/run.log',identity)
        flow=Dataflow(static).run(raw)
        (dest/name/'events.json').write_text(json.dumps(flow.events,ensure_ascii=False,indent=2),encoding='utf-8')
        if name in {'unsupportedArray','missingStringValue'}:
            if not flow.gaps:raise AssertionError('Unsupported behavior was silently certified: '+name)
        elif flow.gaps:raise AssertionError(name+': '+str(flow.gaps))
        if name=='loopWrites':
            writes=[e for e in flow.events if e['domain'][1]=='loop' and e['opcode'] in {54,60,61,62,132}]
            versions=[v for e in writes for v in e['def_ids']]
            assert len(versions)>=7 and len(set(versions))==len(versions)
        if name=='recursiveCalls':
            frames={e['frame_id'] for e in flow.events if e['domain'][1]=='recursive'};assert len(frames)==4
        if name=='heapAlias':assert any(role=='field_read' for _,_,role in flow.links)
        if name=='assignmentChain':
            definitions={v.id:v.definition for v in flow.values}
            loads=[e for e in flow.events if e['domain'][1]=='chain' and e['opcode'] in {27,28}]
            assert len(loads)>=2 and all(e['use_ids'] and all(definitions[v]<e['event_index'] for v in e['use_ids']) for e in loads)
        if name=='falseBranch':assert not any(e['domain'][1]=='chain' for e in flow.events)
        assertions=[e for e in flow.events if e.get('assertion')]
        assert assertions and assertions[-1]['assertion_passed']==(name!='assertionFailure')
        if name=='optimizedLine':assert not any(e['domain'][1]=='optimized' and e['line']==36 for e in flow.events)
        results.append({'fixture':name,'synthetic':True,'real_execution':True,'parity':record['parity'],'matches_expected':record['matches_historical_outcome'],
                        'events':len(raw),'dynamic_uses':len(flow.links),'gaps':flow.gaps,'status':'passed'})
        print(name,'PASS',len(raw),'events',flush=True)
    # A deliberately tiny producer budget must never turn interruption into PASS.
    tiny=dest/'event_limit';tiny.mkdir(exist_ok=True)
    cp=os.pathsep.join(map(str,[build/'main',build/'tests',classes,producer/'lib/*']))
    identity='phase03fixture.Phase03FixtureTest.loopWrites'
    args=[java/'java.exe','-Xmx512m','-noverify','-Dcwm.maxEvents=8',f'-javaagent:{agent}=instrumentingclass=phase03fixture.Phase03Program:phase03fixture.Phase03FixtureTest,logfile=single.log,project=fixture',
          '-cp',cp,'Phase03TestRunner','phase03fixture.Phase03FixtureTest','loopWrites',tiny/'result.tsv']
    result=command(args,tiny,tiny/'run.log',30)
    assert result['exit_code']==86 and completion(tiny/'result.tsv',identity)['outcome']=='UNKNOWN'
    results.append({'fixture':'event_limit','synthetic':True,'real_execution':True,'exit_code':86,'outcome':'UNKNOWN','status':'passed'})
    write_json(ROOT/'outputs/phase03_java_fixtures.json',{'cases':results,'passed':len(results),'benchmark_results':False})

if __name__=='__main__':main()
