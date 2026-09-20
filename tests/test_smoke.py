from mas.evaluate.smoke import run_smoke


def test_smoke_does_not_require_calibration_or_source_data(tmp_path):
    report = run_smoke(tmp_path)
    assert report["status"] == "PASS"
    assert (tmp_path / "smoke_report.json").exists()
