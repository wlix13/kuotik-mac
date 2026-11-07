from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import pefile

MASK_HINTS = {"#0.000": "±0.000", "0.000": "0.000", "   .  ": "число"}


class Reader:
    def __init__(self, b: bytes):
        self.b, self.p = b, 0

    def u8(self) -> int:
        self.p += 1
        return self.b[self.p - 1]

    def take(self, n: int) -> bytes:
        self.p += n
        return self.b[self.p - n:self.p]

    def sstr(self) -> str:
        return self.take(self.u8()).decode("cp1251")

    def value(self):
        t = self.u8()
        if t == 0:
            return None
        if t == 1:
            out = []
            while self.b[self.p] != 0:
                out.append(self.value())
            self.p += 1
            return out
        if t == 2:
            return struct.unpack("<b", self.take(1))[0]
        if t == 3:
            return struct.unpack("<h", self.take(2))[0]
        if t == 4:
            return struct.unpack("<i", self.take(4))[0]
        if t == 5:
            m, e = struct.unpack("<QH", self.take(10))
            s = -1 if e & 0x8000 else 1
            e &= 0x7FFF
            return 0.0 if m == 0 else s * m * 2.0 ** (e - 16383 - 63)
        if t in (6, 7, 8):
            if t == 8:
                return False
            return self.sstr()
        if t == 9:
            return True
        if t == 10:
            n = struct.unpack("<i", self.take(4))[0]
            self.take(n)
            return f"<binary {n}>"
        if t == 11:
            items = []
            while True:
                s = self.sstr()
                if not s:
                    return items
                items.append(s)
        if t == 12:
            n = struct.unpack("<i", self.take(4))[0]
            return self.take(n).decode("cp1251")
        if t == 13:
            return None
        if t == 14:
            items = []
            while self.b[self.p] != 0:
                if self.b[self.p] in (2, 3, 4):
                    self.value()
                assert self.u8() == 1
                items.append(self.props())
            self.p += 1
            return [dict(i) for i in items]
        if t == 15:
            return struct.unpack("<f", self.take(4))[0]
        if t == 16:
            return struct.unpack("<q", self.take(8))[0] / 10000
        if t == 17:
            return struct.unpack("<d", self.take(8))[0]
        if t == 18:
            n = struct.unpack("<i", self.take(4))[0]
            return self.take(n * 2).decode("utf-16-le")
        if t == 19:
            return struct.unpack("<q", self.take(8))[0]
        if t == 20:
            n = struct.unpack("<i", self.take(4))[0]
            return self.take(n).decode("utf-8")
        raise ValueError(f"DFM type {t} at {self.p}")

    def props(self) -> list[tuple[str, object]]:
        out = []
        while True:
            name = self.sstr()
            if not name:
                return out
            out.append((name, self.value()))

    def component(self):
        flags = self.b[self.p]
        if flags & 0xF0 == 0xF0:
            self.p += 1
            if flags & 2:
                self.value()
        cls, name = self.sstr(), self.sstr()
        props = dict(self.props())
        kids = []
        while self.b[self.p] != 0:
            kids.append(self.component())
        self.p += 1
        return cls, name, props, kids


@dataclass
class Node:
    cls: str
    name: str
    props: dict = field(default_factory=dict)
    kids: list["Node"] = field(default_factory=list)

    def get(self, key, default=None):
        return self.props.get(key, default)

    @property
    def caption(self) -> str:
        return str(self.props.get("Caption") or self.props.get("Text") or "")

    def walk(self):
        yield self
        for k in self.kids:
            yield from k.walk()

    def to_dict(self) -> dict:
        return {"cls": self.cls, "name": self.name, "props": self.props,
                "kids": [k.to_dict() for k in self.kids]}

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(d["cls"], d["name"], d["props"], [cls.from_dict(k) for k in d["kids"]])


def _node(component) -> Node:
    cls, name, props, kids = component
    return Node(cls, name, props, [_node(k) for k in kids])


def load_exe(path: str | Path) -> dict[str, Node]:
    pe = pefile.PE(str(path))
    forms: dict[str, Node] = {}
    for t in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        if t.id != 10:
            continue
        for e in t.directory.entries:
            ent = e.directory.entries[0].data.struct
            data = pe.get_data(ent.OffsetToData, ent.Size)
            if data[:4] == b"TPF0":
                r = Reader(data)
                r.p = 4
                node = _node(r.component())
                forms[node.name] = node
    return forms


def save_json(forms: dict[str, Node], path: str | Path) -> None:
    Path(path).write_text(json.dumps({k: v.to_dict() for k, v in forms.items()},
                                     ensure_ascii=False, indent=1), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Node]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: Node.from_dict(v) for k, v in raw.items()}


def tabs(node: Node) -> list[Node]:
    return [k for k in node.walk() if k.cls == "TTabSheet"]
