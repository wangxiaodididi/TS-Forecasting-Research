"""Non-overwrite and source registry integration tests (offline)."""
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

PACKAGE=Path(__file__).resolve().parents[1]
script=PACKAGE/'phase2/install.py'
spec=importlib.util.spec_from_file_location('phase2_install_test',script)
ins=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ins)

BASE_CFG="""MODEL_FILES = {
    'GTR': None,
    'GTRLinearResidual': 'linear_residual.yaml',
    'GTRHaarHigh': 'haar_high.yaml',
    'GTRHaarLow': 'haar_low.yaml',
    'GTRHaarDual': 'haar_dual.yaml',
}
"""
BASE_EXT="""MODELS=['GTRLinearResidual','GTRHaarLow','GTRHaarHigh','GTRHaarDual']
def install_snapshot_extension(snapshot):
    names = ['GTRSignalCore', *MODELS]
"""

class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'lab').mkdir()
        (self.root/'lab/config.py').write_text(BASE_CFG)
        (self.root/'lab/extend.py').write_text(BASE_EXT)
        (self.root/'dataset').mkdir()
        (self.root/'dataset/dont_touch.csv').write_text('abc')
        (self.root/'runs').mkdir()
        (self.root/'runs/historical.txt').write_text('protected')

    def tearDown(self):self.temp.cleanup()

    def test_install_backup_no_overwrite_and_idempotent(self):
        ins.install(self.root,PACKAGE,dry_run=True)
        self.assertFalse((self.root/'phase2').exists())
        ins.install(self.root,PACKAGE)
        cfg=(self.root/'lab/config.py').read_text()
        ext=(self.root/'lab/extend.py').read_text()
        self.assertIn("'GTRPhaseInnovation': 'phase_innovation.yaml'",cfg)
        self.assertIn("'GTRPhaseAdaptive'",ext)
        self.assertIn("'GTRPhaseCore'",ext)
        self.assertEqual((self.root/'runs/historical.txt').read_text(),'protected')
        self.assertEqual((self.root/'dataset/dont_touch.csv').read_text(),'abc')
        ins.install(self.root,PACKAGE)
        self.assertEqual((self.root/'lab/config.py').read_text(),cfg)
        self.assertEqual((self.root/'lab/extend.py').read_text(),ext)

    def test_modified_extension_refuses_overwrite(self):
        ins.install(self.root,PACKAGE)
        (self.root/'extensions/models/GTRPhaseCore.py').write_text('user modifications')
        with self.assertRaises(FileExistsError):
            ins.install(self.root,PACKAGE)

if __name__=='__main__':unittest.main()
