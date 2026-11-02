"""Normal-flow postdominance and explicit source-map uncertainty."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import re
from hashlib import sha256
import xml.etree.ElementTree as ET

IDENTITY_SCHEMA = 'program_identity_v1'
IDENTITY_TYPES = ('SOURCE_BACKED', 'COMPILER_GENERATED', 'UNRESOLVED')


def identity_digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()


def instruction_identity_id(locator):
    return 'instruction-v1:'+identity_digest(locator)


def event_identity_id(lineage, trace_hash, event, instruction_id):
    return 'event-v1:'+identity_digest([lineage,trace_hash,event['event_index'],event['thread_id'],event['frame_id'],instruction_id])


def control_dependence(successors:dict[int,set[int]]) -> set[tuple[int,int]]:
    """A controls B iff B postdominates some but not all successors of A."""
    graph={k:set(v) for k,v in successors.items()};graph[-1]=set()
    if any(v not in graph for values in graph.values() for v in values):raise ValueError("CFG successor missing")
    reaches={-1}
    while True:
        new=reaches|{k for k,v in graph.items() if v&reaches}
        if new==reaches:break
        reaches=new
    if reaches!=set(graph):raise ValueError("CFG contains nodes without a path to exit; no invented postdominator")
    pd={k:({-1} if k==-1 else set(graph)) for k in graph}
    while True:
        new={k:({-1} if k==-1 else {k}|set.intersection(*(pd[v] for v in graph[k]))) for k in graph}
        if new==pd:break
        pd=new
    return {(a,b) for a,ss in graph.items() if len(ss)>1 for b in set.union(*(pd[s] for s in ss))
            if b!=-1 and b not in set.intersection(*(pd[s] for s in ss))}


def load_static(path:Path):
    methods=defaultdict(dict);handlers=defaultdict(list);fields={};allocations={};unsupported=set()
    classes={};method_flags={};calls={};exception_tables={};control_schema=None
    for line in path.read_text(encoding="utf-8").splitlines():
        a=line.split("\t");domain=tuple(a[1:4])
        if a[0]=='CONTROL_SCHEMA':control_schema=a[1]
        elif a[0]=='EXCEPTIONTABLE':exception_tables[domain]={'code_length':int(a[4]),'handler_count':int(a[5]),'handlers':[]}
        elif a[0]=='HANDLERDETAIL':
            exception_tables[domain]['handlers'].append(dict(zip(('table_index','start','end','handler'),map(int,a[4:8])),catch_type=a[8]))
        elif a[0]=="I":
            row={"domain":domain,"line":int(a[4]),"bci":int(a[5]),"opcode":int(a[6]),"next":int(a[7]),"successors":set(map(int,a[8].split(',')))}
            methods[domain][row["bci"]]=row
        elif a[0]=="HANDLER":handlers[domain].append(list(map(int,a[4:7])))
        elif a[0]=="UNSUPPORTED":unsupported.add(domain)
        elif a[0]=="FIELD":fields[(*domain,int(a[4]))]=tuple(a[5:8])
        elif a[0]=="NEW":allocations[(*domain,int(a[4]))]=a[5]
        elif a[0]=="CLASS":classes.setdefault(a[1],{})['superclass']=a[2]
        elif a[0]=="CLASSMETA":
            classes.setdefault(a[1],{}).update(source_file=a[2],access_flags=int(a[3]),major_version=int(a[4]),class_sha256=a[5],class_file=a[6])
        elif a[0]=="METHODMETA":method_flags[domain]=int(a[4])
        elif a[0]=="CALL":calls[(*domain,int(a[4]))]=tuple(a[5:8])
    controls={};gaps={}
    for domain,rows in methods.items():
        try:
            if domain in unsupported:raise ValueError("unsupported legacy subroutine")
            controls[domain]=control_dependence({b:r["successors"] for b,r in rows.items()})
        except ValueError as e:gaps['|'.join(domain)]=str(e)
    return {"methods":dict(methods),"handlers":dict(handlers),"fields":fields,"allocations":allocations,
            "controls":controls,"gaps":gaps,"classes":classes,"method_flags":method_flags,"calls":calls,
            'control_schema':control_schema,'exception_tables':exception_tables}


def mapping(variant:Path, static:dict, features:dict, source_evidence=None):
    """Only XML location fields; no fault marker/label attributes are retained."""
    roles=json.loads((variant/"roles.meta").read_text(encoding="utf-8"))
    selected={name for name,value in features.items() if value}
    if set(roles['features'])!=selected:raise ValueError("roles.meta and configuration features disagree")
    positions={};xml_positions=set()
    for path in sorted((variant/"coverage").glob("spectrum_*_coverage.xml")):
        for package in ET.parse(path).getroot().findall('project/package'):
            package_name=package.get('name','')
            for file in package.findall('file'):
                basename=Path(file.get('path','').replace('\\','/')).stem
                clazz=(package_name+'.' if package_name else '')+basename
                for node in file.findall('line'):
                    if not node.get('featureClass') or not node.get('featureLineNum'):continue
                    composed=(clazz,int(node.get('num')));feature,original_class=node.get('featureClass').split('.',1)
                    value=(feature,original_class,int(node.get('featureLineNum')))
                    if composed in positions and positions[composed]!=value:raise ValueError("Ambiguous original statement mapping")
                    positions[composed]=value;xml_positions.add(composed)
    reverse=defaultdict(set)
    for original,by_feature in roles['roles'].items():
        for feature,composed in by_feature.items():reverse[(feature,composed)].add(original)
    rows=[];by_instruction={};executable=set()
    for domain,instructions in sorted(static['methods'].items()):
        clazz,method,signature=domain
        if source_evidence is not None:
            if clazz not in source_evidence.product_classes(variant):continue
        elif not (variant/'src'/Path(clazz.replace('.','/')).name).exists():
            # Source trees commonly omit package folders; locate by exact basename.
            if not list((variant/'src').rglob(clazz.rsplit('.',1)[-1]+'.java')):continue
        for bci,inst in sorted(instructions.items()):
            loc=(clazz,inst['line']);executable.add(loc);original=positions.get(loc)
            canonical=None;confidence='unmapped';feature=None;reason='no_original_location'
            if original:
                feature,original_class,original_line=original
                names=reverse.get((feature,method),set())
                original_method=next(iter(names)) if len(names)==1 else '<init>' if method=='<init>' else None
                source=variant.parent.parent/'features'/feature/(original_class.replace('.','/')+'.java')
                if not source.exists():
                    alternatives=list((variant.parent.parent/'features'/feature).rglob(original_class.rsplit('.',1)[-1]+'.java'))
                    source=alternatives[0] if len(alternatives)==1 else source
                if feature in selected and original_method and source.is_file() and 0<original_line<=len(source.read_text(encoding='utf-8').splitlines()):
                    canonical='|'.join((feature,clazz,original_method+signature,str(original_line),'0'))
                    confidence='xml_roles_line_checked';reason='line_granularity_ordinal_0_not_AST_statement_identity'
            row={'composed_class':clazz,'composed_method':method,'descriptor':signature,'composed_line':inst['line'],'bytecode_offset':bci,
                 'canonical_statement_id':canonical,'mapping_confidence':confidence,'mapping_note':reason,'feature':feature if canonical else None,
                 'original':list(original) if original else None,'executable':True}
            rows.append(row);by_instruction[(*domain,bci)]=row
    for clazz,line in sorted(xml_positions-executable):
        rows.append({'composed_class':clazz,'composed_line':line,'executable':False,'status':'unmapped_or_non_executable',
                     'reason':'absent_from_compiled_line_table; no dynamic event fabricated','original':list(positions[(clazz,line)])})
    if source_evidence is not None:
        source_evidence.refine(variant, static, selected, reverse, rows)
    return rows,by_instruction,{'configuration_matches_roles':True,'constraints_verified':False,'feature_model_status':'no_version_matched_model_available'}


# Strings/chars are indivisible tokens. Whitespace removal alone is not a proof:
# e.g. "a b" and "ab" must never match. Comments never become mapping evidence.
JAVA_TOKEN = re.compile(r'//[^\r\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][\w$]*|\d+(?:[\w.]*\w)?|>>>?=?|<<=?|==|!=|<=|>=|&&|\|\||\+\+|--|[+*/%&|^!-]=|\S')


def java_tokens(text, line=1):
    if re.search(r'\\u+[0-9a-fA-F]{4}', text):
        raise ValueError('Java Unicode escape preprocessing not supported by source matcher')
    return [(m.group(), line + text.count('\n', 0, m.start())) for m in JAVA_TOKEN.finditer(text)
            if not m.group().startswith(('//', '/*'))]


def source_descriptor(member, classes):
    """Resolve a deliberately bounded Java type subset; ambiguity stays unknown."""
    primitive = dict(zip(('void','boolean','byte','char','short','int','long','float','double'), 'VZBCSIJFD'))
    def dtype(name):
        name = re.sub(r'\s+', '', name); prefix = ''
        while name.endswith('[]'):prefix += '['; name = name[:-2]
        if name in primitive:return prefix + primitive[name]
        package = member['class'].rsplit('.', 1)[0] if '.' in member['class'] else ''
        candidates = {c for c in classes if name == c or c == (package + '.' if package else '') + name}
        if name in {'String','Object'}:candidates.add('java.lang.' + name)
        if len(candidates) != 1:return None
        return prefix + 'L' + next(iter(candidates)).replace('.', '/') + ';'
    args = [dtype(t) for t in member['params'].split(',')] if member['params'] else []
    result = dtype(member['type'])
    return '(' + ''.join(args) + ')' + result if result and all(args) else None


class SourceEvidence:
    """JDK parse-tree spans + exact tokens, not inferred line offsets or labels.

    The JSON contains hashes and parse-tree records only. The exporter never
    compiles the input. Whole method bodies must match after recorded FeatureHouse
    wrapper-call substitution; fields require an exact declaration-token match.
    """
    def __init__(self, path:Path):
        data=json.loads(path.read_text(encoding='utf-8'));self.records=data['records'];self.text={}
        self.digest=sha256(path.read_bytes()).hexdigest()
        for name, digest in data['source_sha256'].items():
            payload=Path(name).read_bytes()
            if sha256(payload).hexdigest()!=digest:raise ValueError('Source evidence hash mismatch: '+name)
            self.text[str(Path(name).resolve())]=payload.decode('utf-8')
        for r in self.records:
            r['path']=str(Path(r['path']).resolve())
            if r['path'] not in self.text:raise ValueError('Unbound source evidence')
            if not 0 <= r['start'] < r['end'] <= len(self.text[r['path']].encode('utf-16-le'))//2:
                raise ValueError('Invalid compiler source span')

    def members(self, root):
        return [r for r in self.records if Path(r['path']).is_relative_to(root.resolve())]

    def product_classes(self, variant):
        return {r['class'] for r in self.members(variant/'src') if r['kind']=='class'}

    def tokens(self, r, body=False, wrappers=()):
        a,b=(r['body_start'],r['body_end']) if body else (r['start'],r['end'])
        if a<0:return []
        raw=self.text[r['path']].encode('utf-16-le');prefix=raw[:2*a].decode('utf-16-le')
        tokens=java_tokens(raw[2*a:2*b].decode('utf-16-le'), prefix.count('\n')+1)
        return [('original' if t in wrappers and i+1<len(tokens) and tokens[i+1][0]=='(' else t,l)
                for i,(t,l) in enumerate(tokens)]

    def refine(self, variant, static, selected, reverse, rows):
        product=self.members(variant/'src');feature_root=variant.parent.parent/'features'
        originals=[r for r in self.members(feature_root) if Path(r['path']).relative_to(feature_root).parts[0] in selected]
        classes={r['class'] for r in product+originals if r['kind']=='class'}
        wrappers={name for (_,name) in reverse if '__wrappee__' in name}
        methods={};fields={};proofs={}
        for p in product:
            if p['kind'] not in {'method','field'}:continue
            if p['kind']=='field':
                matches=[r for r in originals if r['kind']=='field' and r['class']==p['class'] and r['name']==p['name']
                         and [t for t,l in self.tokens(r)]==[t for t,l in self.tokens(p)]]
                fields.setdefault((p['class'],p['line']),[]).append((p,matches))
                continue
            descriptor=source_descriptor(p, classes)
            if descriptor is None:continue
            domain=(p['class'],p['name'],descriptor);methods[domain]=p
            # Explicit composition annotation must identify one exact feature AST span.
            before=self.text[p['path']].encode('utf-16-le')[:2*p['start']].decode('utf-16-le')
            marker=re.search(r'//__feature_mapping__\s+\[([^\]]+)\]\s+\[(\d+):(\d+)\]\s*$', before)
            if not marker:continue
            feature,start,end=marker[1],int(marker[2]),int(marker[3])
            names=reverse.get((feature,p['name']),set()) if p['name']!='<init>' else {'<init>'}
            matches=[r for r in originals if r['kind']=='method' and r['class']==p['class'] and r['name'] in names
                     and Path(r['path']).relative_to(feature_root).parts[0]==feature
                     and (r['line'],r['end_line'])==(start,end) and source_descriptor(r,classes)==descriptor
                     and r.get('parameter_names')==p.get('parameter_names')]
            if len(matches)!=1:continue
            original=matches[0];a=self.tokens(p,True,wrappers);b=self.tokens(original,True)
            if not a or [t for t,l in a]!=[t for t,l in b]:continue
            line_map=defaultdict(set)
            for (_,x),(_,y) in zip(a,b):line_map[x].add(y)
            # Header is a declaration location, never an AST statement ordinal.
            for n in range(p['line'],a[0][1]+1):line_map[n].add(original['line'])
            proofs[domain]=(p,original,line_map)
        for row in rows:
            clazz,line=row['composed_class'],row['composed_line']
            line_verified=any(p['kind']=='class' and p['class']==clazz and p['line']<=line<=p['end_line'] for p in product)
            row['composed_source_line_id']='product-line|'+clazz+'|'+str(line) if line_verified else None
            row['composed_source_line_verified']=line_verified
            row['source_statement_verified']=False;row['statement_ordinal']=None
            row['mapping_granularity']='source_line';row['source_evidence_sha256']=self.digest
            if not row['executable']:
                declarations=[p for p in product if p['class']==clazz and p['kind']=='method' and p['line']<=line
                              and p['body_start']>=0 and line<=self.text[p['path']].encode('utf-16-le')[:2*p['body_start']].decode('utf-16-le').count('\n')+1]
                tries=[p for p in product if p['class']==clazz and p['kind']=='try' and p['line']==line]
                reason='method_declaration_without_separate_instruction' if declarations else 'try_keyword_without_separate_instruction' if tries else 'UNKNOWN_absent_line_table_reason'
                row.update(status='non_instruction_source_anchor' if declarations or tries else 'unmapped_or_non_executable',reason=reason,
                           classification_evidence=[{'path':p['path'],'kind':p['kind'],'start_line':p['line'],'end_line':p['end_line']} for p in declarations+tries],
                           no_dynamic_event_expected=bool(declarations or tries))
                continue
            domain=(clazz,row['composed_method'],row['descriptor']);inst=static['methods'][domain][row['bytecode_offset']]
            row['legacy_mapping_confidence']=row['mapping_confidence']
            row.update(canonical_statement_id=None,canonical_source_line_id=None,feature=None,original=None,
                       mapping_confidence='unmapped',mapping_note='UNKNOWN_no_unique_source_proof',source_construct='unknown')
            candidates=[];kind=None;reason=None
            field=fields.get((clazz,line))
            if domain[1] in {'<init>','<clinit>'} and field and len(field)==1 and inst['opcode']!=177:
                p,matches=field[0]
                # Field initialization must contain an actual qualified store in this method/line.
                stores=[static['fields'].get((*domain,bci)) for bci,x in static['methods'][domain].items()
                        if x['line']==line and x['opcode'] in {179,181}]
                field_signature=source_descriptor({**p,'params':''},classes)
                field_type=field_signature.split(')',1)[1] if field_signature else None
                if field_type and (clazz,p['name'],field_type) in stores and p['body_start']>=0:
                    candidates=[(r,r['line'],'field:'+r['name']+':'+r['type']) for r in matches]
                    kind='field_initializer';reason='exact_field_declaration_tokens_and_qualified_bytecode_store'
            elif domain in proofs:
                p,original,line_map=proofs[domain]
                if len(line_map.get(line,set()))==1:
                    n=next(iter(line_map[line]));candidates=[(original,n,original['name']+domain[2])]
                    kind='method_body_line';reason='annotation_AST_signature_and_exact_body_token_alignment'
                    if line==p['end_line'] and inst['opcode']==177:kind='implicit_return_at_explicit_method_end'
                    if any(r['class']==clazz and r['kind']=='finally' and r['end_line']==line for r in product):kind='compiler_finally_epilogue_line'
                    if domain[1]=='<init>' and line==p['line']:kind='implicit_super_call_at_explicit_constructor'
            if len(candidates)==1:
                r,n,member=candidates[0];feature=Path(r['path']).relative_to(feature_root).parts[0]
                identity='|'.join(('line',feature,clazz,member,str(n)))
                row.update(canonical_statement_id=identity,canonical_source_line_id=identity,feature=feature,
                           original=[feature,clazz,n],mapping_confidence='AST_tokens_verified_source_line',mapping_note=reason,
                           source_construct=kind,source_path=r['path'],source_member=member)
            elif len(candidates)>1:
                row['mapping_note']='UNKNOWN_ambiguous_equal_feature_members'
            elif field and len(field)>1:
                row['mapping_note']='UNKNOWN_multiple_field_declarations_on_same_line'
            elif domain[1]=='<init>' and not any(p['kind']=='method' and p['class']==clazz and p['name']=='<init>' for p in product):
                row.update(mapping_note='UNKNOWN_no_explicit_source_member_for_default_constructor',source_construct='implicit_default_constructor')


def program_identities(variant, static, mapped, source, binding, required=(), *, artifact_resolver=None):
    """Certify identities from unchanged class bytes, bound sources and parse AST.

    Unknown locations never become generated merely because they lack a source
    map. Only the proven default-constructor prologue/return subset is supported.
    Field initializers keep their existing SOURCE_BACKED proof, even when inside
    an implicit constructor. No graph nodes or execution events are fabricated.
    """
    # Operational paths may differ on a verified export. Evidence strings and
    # binding digests stay unchanged; the resolver must verify its own receipt.
    resolve_artifact=artifact_resolver or Path
    product=source.members(variant/'src');tests=source.members(variant/'test')
    members=product+tests;known_classes={r['class'] for r in members if r['kind']=='class'}
    checked={};defaults={};out={}
    for clazz in {k[0] for k in set(mapped)|set(required)}:
        c=static.get('classes',{}).get(clazz,{})
        candidates=[r for r in members if r['kind']=='class' and r['class']==clazz]
        reason='missing_or_ambiguous_class_AST'
        if len(candidates)==1:
            ast=candidates[0];class_file=c.get('class_file');path=ast['path']
            reason='unbound_class_or_source_artifact'
            if (class_file and c.get('source_file')==Path(path.replace('\\','/')).name and
                binding.get('class_sha256',{}).get(class_file)==c.get('class_sha256') and
                binding.get('source_sha256',{}).get(path)==sha256(resolve_artifact(path).read_bytes()).hexdigest() and
                sha256(resolve_artifact(class_file).read_bytes()).hexdigest()==c.get('class_sha256')):
                checked[clazz]=(ast,c,None)
                continue
        checked[clazz]=(None,c,reason)
    for domain,instructions in static['methods'].items():
        clazz,method,desc=domain
        if method!='<init>' or clazz not in checked:continue
        ast,c,error=checked[clazz]
        if error:continue
        explicit=[r for r in members if r['class']==clazz and r['kind']=='method' and r['name']=='<init>']
        flags=static.get('method_flags',{}).get(domain)
        ordered=sorted(instructions);ops=[instructions[b]['opcode'] for b in ordered]
        if (explicit or desc!='()V' or ast.get('class_kind')!='CLASS' or
            ast.get('class_nesting') not in {'TOP_LEVEL','STATIC_MEMBER'} or flags is None or flags & (8|256|1024) or
            c.get('superclass')!='java.lang.Object' or static['handlers'].get(domain) or len(ordered)<3):continue
        if (ops[:2]!=[42,183] or ops[-1]!=177 or
            static.get('calls',{}).get((*domain,ordered[1]))!=('java.lang.Object','<init>','()V')):continue
        # Every middle instruction must already have a checked explicit field origin.
        if any(mapped.get((*domain,b),{}).get('source_construct')!='field_initializer' or
               mapped[(*domain,b)].get('mapping_confidence')!='AST_tokens_verified_source_line' for b in ordered[2:-1]):continue
        if any(instructions[b]['successors']!={ordered[i+1] if i+1<len(ordered) else -1} for i,b in enumerate(ordered)):continue
        roles={ordered[0]:'default_constructor_receiver',ordered[1]:'default_constructor_super_call',ordered[-1]:'default_constructor_return'}
        defaults[domain]={'roles':roles,'source_constructor_count':0,'constructor_descriptor':desc,
                          'super_call':['java.lang.Object','<init>','()V'],'method_access_flags':flags,
                          'acc_synthetic':bool(flags & 4096),'linear_body_verified':True,'class_kind':ast['class_kind'],
                          'class_nesting':ast['class_nesting'],'field_initializer_instruction_count':len(ordered)-3}
    for where in sorted(set(mapped)|set(required)):
        domain=where[:3];bci=where[3];clazz,method,desc=domain
        row=mapped.get(where);inst=static['methods'].get(domain,{}).get(bci)
        ast,c,error=checked.get(clazz,(None,{},'missing_class_evidence'))
        locator={'class_sha256':c.get('class_sha256'),'declaring_class':clazz,'method':method,'descriptor':desc,'bytecode_offset':bci}
        proof={'source_evidence_sha256':source.digest,'compilation_binding_sha256':identity_digest(binding),
               'class_file':c.get('class_file'),'class_major_version':c.get('major_version'),
               'source_file':ast['path'] if ast else None,
               'source_sha256':binding.get('source_sha256',{}).get(ast['path']) if ast else None}
        result={'schema':IDENTITY_SCHEMA,'provenance_type':'UNRESOLVED','scope':'product' if row else 'test',
                'instruction_id':None,'locator':locator,'opcode':inst['opcode'] if inst else None,
                'generation_kind':None,'source_line_id':row.get('canonical_source_line_id') if row else None,
                'source_granularity':'source_line','source_statement_verified':False,
                'evidence':proof,'unresolved_reason':error or 'no_verified_source_or_generated_rule',
                'alignment':{'level':'none','key':None}}
        if not error and inst:
            if row and row.get('canonical_source_line_id') and row.get('mapping_confidence')=='AST_tokens_verified_source_line':
                proof['rule']='AST_feature_tokens_and_bound_class';proof['source_member']=row['source_member']
                proof['feature_source_sha256']=sha256(resolve_artifact(row['source_path']).read_bytes()).hexdigest()
                result['provenance_type']='SOURCE_BACKED'
                result['alignment']={'level':'source_line_candidate_not_instruction_equivalence',
                                     'key':identity_digest([proof['feature_source_sha256'],row['canonical_source_line_id']])}
            elif domain in defaults and bci in defaults[domain]['roles']:
                proof['rule']='AST_absent_constructor_and_verified_default_bytecode'
                proof['default_constructor']={k:v for k,v in defaults[domain].items() if k!='roles'}
                result['provenance_type']='COMPILER_GENERATED';result['generation_kind']=defaults[domain]['roles'][bci]
                result['alignment']={'level':'semantic_role_only_not_node_equivalence','key':result['generation_kind']}
            elif not row or any(r['kind']=='method' and r['class']==clazz and r['name']==method for r in product):
                methods=[r for r in (product if row else tests) if r['class']==clazz and r['kind']=='method' and r['name']==method
                         and source_descriptor(r,known_classes)==desc and r['line']<=inst['line']<=r['end_line']]
                if len(methods)==1:
                    result['provenance_type']='SOURCE_BACKED'
                    result['source_line_id']=('composed-line|' if row else 'test-line|')+clazz+'|'+method+desc+'|'+str(inst['line'])
                    proof['rule']='AST_product_member_and_bound_class' if row else 'AST_test_member_and_bound_class'
                    proof['source_member']=method+desc
        if result['provenance_type']!='UNRESOLVED':
            result['instruction_id']=instruction_identity_id(locator);result['unresolved_reason']=None
        result['evidence_sha256']=identity_digest(proof)
        out[where]=result
        if row:row['program_identity']=result
    return out


def trusted_program_identity(identity):
    """Validate the certificate structure at consumers; never trust a type flag alone.

    Source/class hashes are verified by the producer, not re-read by training.
    This checks artifact consistency, not adversarial cryptographic attestation.
    """
    if not isinstance(identity,dict) or identity.get('schema')!=IDENTITY_SCHEMA:return False
    p=identity.get('provenance_type');e=identity.get('evidence',{});loc=identity.get('locator',{})
    if p not in IDENTITY_TYPES[:2] or identity.get('unresolved_reason') is not None:return False
    if identity.get('instruction_id')!=instruction_identity_id(loc) or identity.get('evidence_sha256')!=identity_digest(e):return False
    for value in (loc.get('class_sha256'),e.get('source_sha256'),e.get('source_evidence_sha256'),e.get('compilation_binding_sha256')):
        if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value):return False
    if (type(loc.get('bytecode_offset')) is not int or loc['bytecode_offset']<0 or
        not all(isinstance(loc.get(k),str) and loc[k] for k in ('declaring_class','method','descriptor')) or
        type(identity.get('opcode')) is not int or identity.get('scope') not in {'product','test'}):return False
    if p=='SOURCE_BACKED':
        return bool(identity.get('source_line_id') and e.get('source_member') and
                    e.get('rule') in {'AST_feature_tokens_and_bound_class','AST_test_member_and_bound_class','AST_product_member_and_bound_class'})
    d=e.get('default_constructor',{})
    return bool(e.get('rule')=='AST_absent_constructor_and_verified_default_bytecode' and
                loc['method']=='<init>' and loc['descriptor']=='()V' and d.get('source_constructor_count')==0 and
                d.get('linear_body_verified') is True and d.get('constructor_descriptor')=='()V' and
                d.get('super_call')==['java.lang.Object','<init>','()V'] and
                d.get('class_kind')=='CLASS' and d.get('class_nesting') in {'TOP_LEVEL','STATIC_MEMBER'} and
                type(d.get('method_access_flags')) is int and not d['method_access_flags'] & (8|256|1024) and
                identity.get('source_line_id') is None and
                {'default_constructor_receiver':42,'default_constructor_super_call':183,'default_constructor_return':177}.get(identity.get('generation_kind'))==identity['opcode'])
