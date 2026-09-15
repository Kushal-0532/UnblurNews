import pytest

from case_logic import _leaning_bucket, _sentiment_polarity, determine_case, dominant_leaning


def _a(pol, sent):
    return {"political_score": pol, "sentiment_score": sent}


@pytest.mark.parametrize("score, expected", [
    (-1.0, "left"), (-0.31, "left"), (-0.3, "center"), (0.0, "center"),
    (0.3, "center"), (0.31, "right"), (1.0, "right"),
])
def test_leaning_bucket_boundaries(score, expected):
    assert _leaning_bucket(score) == expected


@pytest.mark.parametrize("score, expected", [
    (-1.0, "negative"), (-0.21, "negative"), (-0.2, "neutral"), (0.0, "neutral"),
    (0.2, "neutral"), (0.21, "positive"), (1.0, "positive"),
])
def test_sentiment_polarity_boundaries(score, expected):
    assert _sentiment_polarity(score) == expected


def test_determine_case_empty_is_balanced():
    assert determine_case([]) == "balanced"


def test_determine_case_echo_chamber():
    articles = [_a(-0.6, -0.5)] * 8 + [_a(0.1, 0.1)]
    assert determine_case(articles, current_political=-0.6, current_sentiment=-0.5) == "echo_chamber"


def test_determine_case_contradiction():
    articles = [_a(-0.8, -0.4), _a(-0.7, -0.3), _a(0.8, 0.4), _a(0.7, 0.3)]
    assert determine_case(articles) == "contradiction"


def test_determine_case_internal_split():
    articles = [_a(-0.5, -0.8), _a(-0.6, -0.7), _a(-0.4, 0.8), _a(-0.5, 0.7)]
    assert determine_case(articles, current_political=-0.5) == "internal_split"


def test_determine_case_balanced():
    articles = [_a(-0.5, 0.0), _a(0.0, 0.0), _a(0.5, 0.0), _a(-0.2, 0.1)]
    assert determine_case(articles) == "balanced"


def test_dominant_leaning_empty():
    assert dominant_leaning([]) == ("center", 0.0)


def test_dominant_leaning_majority():
    assert dominant_leaning([_a(0.5, 0), _a(0.6, 0), _a(-0.5, 0)]) == ("right", 66.7)


def test_dominant_leaning_tie_picks_first_in_left_center_right_order():
    assert dominant_leaning([_a(0.5, 0), _a(-0.5, 0)]) == ("left", 50.0)
    assert dominant_leaning([_a(0.5, 0), _a(0.0, 0)]) == ("center", 50.0)
