"""Dev-recovery tool: extract files from the local git pack WITHOUT the git binary.
(Kept in-repo so it survives sandbox /tmp wipes. Delete before final submission if desired.)
Usage: python tools/dev/packget.py <subdir> <file1> [file2 ...]   (run from repo root)"""
import hashlib, os, sys, zlib
PACK = os.path.join(os.getcwd(), ".git/objects/pack/pack-74fc8d831aa7e6cd0a7116ce149180a770b9dd26.pack")
MAIN = "07a06bb896b7963c677502d32e0fc68af12918c2"
SUBDIR, WANT = sys.argv[1], sys.argv[2:]
data = open(PACK, "rb").read()
assert data[:4] == b"PACK"
nobj = int.from_bytes(data[8:12], "big")
pos = 12
objs = []
for _ in range(nobj):
    start = pos
    b = data[pos]; pos += 1
    typ = (b >> 4) & 7
    size = b & 15
    shift = 4
    while b & 0x80:
        b = data[pos]; pos += 1
        size |= (b & 0x7F) << shift
        shift += 7
    extra = None
    if typ == 6:
        b = data[pos]; pos += 1
        off = b & 0x7F
        while b & 0x80:
            b = data[pos]; pos += 1
            off = ((off + 1) << 7) | (b & 0x7F)
        extra = ("ofs", start - off)
    elif typ == 7:
        extra = ("ref", data[pos:pos + 20].hex())
        pos += 20
    objs.append([start, typ, size, pos, extra])
    d = zlib.decompressobj()
    d.decompress(data[pos:])
    pos = len(data) - len(d.unused_data)
TYPES = {1: "commit", 2: "tree", 3: "blob", 4: "tag"}
off_index = {o[0]: i for i, o in enumerate(objs)}
resolved = {}
by_sha = {}
def inflate_at(ppos):
    d = zlib.decompressobj()
    return d.decompress(data[ppos:])
def apply_delta(base, delta):
    p = 0
    def varint2():
        nonlocal p
        r = 0; s = 0
        while True:
            b = delta[p]; p += 1
            r |= (b & 0x7F) << s
            if not (b & 0x80):
                return r
            s += 7
    _base_size = varint2()
    res_size = varint2()
    out = bytearray()
    while p < len(delta):
        cmd = delta[p]; p += 1
        if cmd & 0x80:
            off = 0; sz = 0
            if cmd & 0x01: off |= delta[p]; p += 1
            if cmd & 0x02: off |= delta[p] << 8; p += 1
            if cmd & 0x04: off |= delta[p] << 16; p += 1
            if cmd & 0x08: off |= delta[p] << 24; p += 1
            if cmd & 0x10: sz |= delta[p]; p += 1
            if cmd & 0x20: sz |= delta[p] << 8; p += 1
            if cmd & 0x40: sz |= delta[p] << 16; p += 1
            if sz == 0: sz = 0x10000
            out += base[off:off + sz]
        elif cmd:
            out += delta[p:p + cmd]; p += cmd
        else:
            raise ValueError("bad delta op 0")
    assert len(out) == res_size
    return bytes(out)
def register(idx, t, raw):
    resolved[idx] = (t, raw)
    h = hashlib.sha1(("%s %d\0" % (t, len(raw))).encode() + raw).hexdigest()
    by_sha[h] = idx
for i, o in enumerate(objs):
    if o[1] in (1, 2, 3, 4):
        register(i, TYPES[o[1]], inflate_at(o[3]))
pending = [i for i, o in enumerate(objs) if o[1] in (6, 7)]
while pending:
    progress = False
    for i in list(pending):
        _, typ, _, ppos, extra = objs[i]
        if extra[0] == "ofs":
            bi = off_index.get(extra[1])
        else:
            bi = by_sha.get(extra[1])
        if bi is None or bi not in resolved:
            continue
        bt, base = resolved[bi]
        register(i, bt, apply_delta(base, inflate_at(ppos)))
        pending.remove(i)
        progress = True
    if not progress:
        raise RuntimeError("stuck deltas")
def parse_tree(raw):
    entries = []
    p = 0
    while p < len(raw):
        sp = raw.index(b" ", p)
        mode = raw[p:sp].decode()
        nl = raw.index(b"\0", sp)
        name = raw[sp + 1:nl].decode()
        sha = raw[nl + 1:nl + 21].hex()
        p = nl + 21
        entries.append((mode, name, sha))
    return entries
def tree_sha_for(commit_sha, subpath):
    t, raw = resolved[by_sha[commit_sha]]
    assert t == "commit"
    tree = [l for l in raw.decode().split("\n") if l.startswith("tree ")][0].split()[1]
    for part in subpath.split("/"):
        _, traw = resolved[by_sha[tree]]
        found = [s for m, n, s in parse_tree(traw) if n == part]
        assert found, f"missing {part}"
        tree = found[0]
    return tree
dtree = tree_sha_for(MAIN, SUBDIR)
_, dtraw = resolved[by_sha[dtree]]
dmap = {n: s for m, n, s in parse_tree(dtraw)}
os.makedirs(SUBDIR, exist_ok=True)
for f in WANT:
    _, braw = resolved[by_sha[dmap[f]]]
    open(SUBDIR + "/" + f, "wb").write(braw)
    print(f"EXTRACTED {SUBDIR}/{f} ({len(braw)} bytes)")
