from mas.evaluate.submission_check import check_submission


def test_submission_check_reports_missing_artifacts(tmp_path):
    report = check_submission(tmp_path, project_root=tmp_path)
    assert report["status"] == "INCOMPLETE"
    assert report["missing"]
    assert (tmp_path / "submission_check.json").exists()
