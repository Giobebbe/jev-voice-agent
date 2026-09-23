from jevagent import text as T


def test_sentences_keep_domains_together():
    assert T.sentences("Now can you open up X.com? Nice, nice.") == ["Now can you open up X.com?", "Nice, nice."]


def test_boundaries_need_a_real_left_side():
    assert T.boundaries("and search") == []
    bs = T.boundaries("Open the notes app and create a new note")
    assert [b.right for b in bs] == ["create a new note"]


def test_split_at_cuts_before_connector():
    s = "Take a screenshot and then open the JEV folder"
    bs = T.boundaries(s)
    assert T.split_at(s, bs) == ["Take a screenshot", "open the JEV folder"]
    assert T.split_at(s, []) == [s]


def test_candidate_spans_include_multiword_and_skip_fillers():
    spans = T.candidate_spans("can you Google search Norbert Wiener?")
    assert "Norbert Wiener" in spans
    assert "Wiener" in spans
    assert all(not s.lower().startswith(("um ", "okay ")) for s in spans)
    assert len(spans) == len({s.lower() for s in spans})


def test_candidate_spans_capped():
    spans = T.candidate_spans(" ".join(f"w{i}" for i in range(60)))
    assert len(spans) <= 240


def test_domains():
    assert T.domains("open up X.com please") == ["x.com"]
    assert T.domains("go to x dot com") == ["x.com"]
    assert T.domains("open github.com and news.ycombinator.com") == ["github.com", "news.ycombinator.com"]


def test_names_item():
    assert T.names_item("move the shopping list into notes", "shopping list.txt")
    assert T.names_item("delete project artemis", "Project Artemis/")
    assert not T.names_item("move the mission plan into project apollo", "Notes/Untitled.txt")
    assert not T.names_item("anything", None)
