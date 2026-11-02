"""Artificial containers copied from one authorized TRAIN trace; no target reads."""
import json,os,shutil,tempfile,unittest
from pathlib import Path
from cwm_fp.data.common import sha256
from cwm_fp.models.prefix_dataset import OUT,load
from cwm_fp.models.prefix_reader import load_prefix
SOURCE=Path(os.environ.get('CWM_FP_PREFIX_TEST_SOURCE',str(OUT)))

class PrefixReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.m=load(SOURCE/'prefix_manifest.json');self.m['traces']=self.m['traces'][:1]
        self.assertEqual('train',self.m['traces'][0]['partition'])
        refs=list(self.m['traces'][0]['input_files'].values())+[self.m['framework_static']]
        for r in refs:
            d=self.root/r['path'];d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/r['path'],d)
        self.save()
    def tearDown(self):self.tmp.cleanup()
    def save(self):
        (self.root/'prefix_manifest.json').write_text(json.dumps(self.m),encoding='utf8');self.anchor=sha256(self.root/'prefix_manifest.json')
    def read(self,event=0,anchor=None):return load_prefix(self.root,0,event,expected_manifest_sha256=anchor or self.anchor)
    def test_load_without_target_file(self):
        self.assertFalse((self.root/self.m['traces'][0]['targets']['path']).exists());self.assertGreater(len(self.read().model['x']),0)
    def test_external_anchor_required(self):
        with self.assertRaises(ValueError):self.read(anchor='0'*64)
    def test_changed_observation_refused(self):
        p=self.root/self.m['traces'][0]['input_files']['observations']['path'];p.write_text('tampered',encoding='utf8')
        with self.assertRaises(ValueError):self.read()
    def test_escaping_static_refused(self):
        self.m['traces'][0]['input_files']['static']['path']='../../outside.json';self.save()
        with self.assertRaises(ValueError):self.read()
    def test_test_partition_refused(self):
        self.m['traces'][0]['partition']='test';self.save()
        with self.assertRaises(ValueError):self.read()
    def test_unknown_anchor_not_normal_end(self):
        with self.assertRaises(ValueError):self.read(10**8)

if __name__=='__main__':unittest.main()
