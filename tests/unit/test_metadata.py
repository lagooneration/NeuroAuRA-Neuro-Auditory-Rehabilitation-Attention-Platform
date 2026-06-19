"""Unit tests for the metadata validator and alignment grading."""

from neurophile.data.validators.metadata import validate_session, _compute_alignment_grade


def test_valid_session_grade_b(bids_root):
    """A session with LSL sync and offset < 3 ms should get Grade B."""
    result = validate_session(bids_root, "01", "01", "aad")
    assert result.is_valid, f"Expected valid but got errors: {result.errors}"
    assert result.alignment_grade == "B"


def test_missing_sidecar_returns_grade_f(tmp_path):
    """A session with no sidecar JSON should return Grade F."""
    result = validate_session(tmp_path, "99", "01", "aad")
    assert not result.is_valid
    assert result.alignment_grade == "F"


def test_grade_a_ttl_sync():
    sidecar = {
        "SamplingFrequency": 512,
        "EEGReference": "average",
        "PowerLineFrequency": 50,
        "HardwareFilters": {},
        "SoftwareFilters": {},
        "StimulusSyncMethod": "TTL",
        "MeasuredSyncOffset_ms": 0.3,
        "MeasuredDrift_ppm": 0.5,
    }
    grade = _compute_alignment_grade(sidecar, errors=[])
    assert grade == "A"


def test_grade_d_software_only():
    sidecar = {"StimulusSyncMethod": "software_only"}
    grade = _compute_alignment_grade(sidecar, errors=[])
    assert grade == "D"


def test_grade_f_with_errors():
    grade = _compute_alignment_grade({}, errors=["something wrong"])
    assert grade == "F"


def test_ci_warning_when_field_missing(bids_root):
    """Session without CochlearImplant field should produce an info message, not an error."""
    result = validate_session(bids_root, "01", "01", "aad")
    ci_mentioned = any("CochlearImplant" in m or "normal-hearing" in m for m in result.info)
    assert ci_mentioned
