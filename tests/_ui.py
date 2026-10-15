"""Look at the page a Shiny tab draws, instead of at the source text that draws it.

``page(ui_function)`` renders the page of a tab (a Shiny module UI, given an ``id``) and
returns a small tree whose elements know their parent, so that a test can ask what the user
would see: which control comes before which, which are side by side, how wide the columns are.
A test of this kind survives any refactoring of the code that does not change the page.
"""

from __future__ import annotations

from html.parser import HTMLParser

_VOID = {"input", "br", "img", "meta", "link", "hr", "source", "area", "base", "col"}
NAMESPACE = "t"


class Node:
    def __init__(self, tag, attrs, parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children: list[Node] = []
        self.parts: list = []  # text and child nodes in order
        self.order = 0

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def text(self) -> str:
        return " ".join(
            p.strip() if isinstance(p, str) else p.text() for p in self.parts
        ).strip()

    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    def find_all(self, predicate):
        return [n for n in self.walk() if predicate(n)]


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root", {})
        self.current = self.root
        self.count = 0

    def handle_starttag(self, tag, attrs):
        node = Node(tag, [(k, v or "") for k, v in attrs], self.current)
        self.count += 1
        node.order = self.count
        self.current.children.append(node)
        self.current.parts.append(node)
        if tag not in _VOID:
            self.current = node

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.current = self.current.parent

    def handle_endtag(self, tag):
        node = self.current
        while node is not None and node.tag != tag:
            node = node.parent
        if node is not None and node.parent is not None:
            self.current = node.parent

    def handle_data(self, data):
        self.current.parts.append(data)


class Page:
    """The tree of one rendered page, with questions a user could ask of it."""

    def __init__(self, html: str):
        builder = _Builder()
        builder.feed(html)
        self.root = builder.root

    def element(self, name: str) -> Node:
        """The element of a control or an output, by the id the code gave it."""
        wanted = {name, f"{NAMESPACE}-{name}"}
        for node in self.root.walk():
            if node.attrs.get("id") in wanted:
                return node
        raise AssertionError(f"the page has nothing with the id {name!r}")

    def has(self, name: str) -> bool:
        try:
            self.element(name)
        except AssertionError:
            return False
        return True

    def comes_before(self, *names: str) -> bool:
        """Whether the elements of these ids are in this order on the page."""
        orders = [self.element(n).order for n in names]
        return orders == sorted(orders) and len(set(orders)) == len(orders)

    def heading(self, text: str) -> Node:
        """The card header or the panel title with exactly this text."""
        for node in self.root.walk():
            if node.text() == text and (
                "card-header" in node.classes() or node.tag in ("h3", "h4", "h5")
            ):
                return node
        raise AssertionError(f"the page has no heading {text!r}")

    def card_of(self, node: Node) -> Node:
        for up in node.ancestors():
            if "card" in up.classes():
                return up
        raise AssertionError("not inside a card")

    def inside(self, name: str, container: Node) -> bool:
        return container in self.element(name).ancestors()

    def columns(self, name: str) -> tuple[str, int]:
        """The column widths of the row the element is in, and the column it is in.

        ``("7,5", 1)``: in the second of two columns, a seven to five split.
        """
        node = self.element(name)
        for up in node.ancestors():
            if up.tag == "bslib-layout-columns":
                cell = node
                while cell.parent is not up:
                    cell = cell.parent
                return up.attrs.get("col-widths-sm", ""), up.children.index(cell)
        raise AssertionError(f"{name!r} is not in a row of columns")

    def same_row(self, *names: str) -> bool:
        """Whether the elements are side by side in one row of columns."""
        rows = []
        for name in names:
            for up in self.element(name).ancestors():
                if up.tag == "bslib-layout-columns":
                    rows.append(up)
                    break
            else:
                return False
        return all(r is rows[0] for r in rows)

    def ids(self) -> set[str]:
        return {
            n.attrs["id"].removeprefix(f"{NAMESPACE}-")
            for n in self.root.walk()
            if "id" in n.attrs
        }


def page(ui_function) -> Page:
    """Render a module UI function (called with the namespace id) and return its page."""
    import htmltools

    html = htmltools.TagList(ui_function(NAMESPACE)).render()["html"]
    return Page(html)
