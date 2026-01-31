# src/tests/test_config.py
from aoe.config import AOEConfig
def test_load_and_override(tmp_path):
    yml = tmp_path/"t.yaml"
    yml.write_text("data:\n  start: '2022-01-01'\n  end: '2022-02-01'\n  universe: ['AAPL']\nstrategy:\n  kind: 'condor'\n  dte: 21\n")
    cfg = AOEConfig.load(str(yml), {"start":"2022-03-01","universe":"MSFT,SPY"})
    assert cfg["data"]["start"] == "2022-03-01"
    assert cfg["data"]["universe"] == ["MSFT","SPY"]
    assert cfg["strategy"]["dte"] == 21
