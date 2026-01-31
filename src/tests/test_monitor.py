from aoe.monitor.early import mark_and_check
from types import SimpleNamespace

class DummyModel:
    sigma0 = 0.25
    def price_euro(self,*a,**k): return 5.0

def test_early_exercise_put():
    mdl = DummyModel()
    res = mark_and_check(mdl,"P",s0=90,k=100,r=0.02,q=0.0,dte=10,long_qty=1)
    assert res["exercise"] and res["mark"]==0.0

def test_hold_call():
    mdl = DummyModel()
    res = mark_and_check(mdl,"C",s0=110,k=100,r=0.02,q=0.0,dte=10,long_qty=1)
    assert not res["exercise"] and res["mark"]>0
