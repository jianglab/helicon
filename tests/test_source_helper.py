from _source import contains, count, squash


def test_the_layout_does_not_matter():
    code = 'x = helicon.shiny.slider(\n        "types_n",\n        1,\n    )\n'
    assert contains(
        code, 'helicon.shiny.slider(\n                            "types_n"'
    )
    assert contains(code, 'slider( "types_n"')
    assert not contains(code, 'slider("other"')


def test_it_counts_whatever_the_layout():
    code = "a(\n  1)\nb\nc(\n        1)\n"
    assert count(code, "c(\n 1)") == 1 and count(code, "a( 1)") == 1
    assert squash("  a \n\t b ") == " a b "
