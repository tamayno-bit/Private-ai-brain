"""Owner-authenticated quarantine of Markdown plans uploaded from Brain Control.

Uploads are UNTRUSTED text: no execution, no learning, no active prompts.
Each upload is immutable, hashed, and stored on the persistent volume.
"""
from pathlib import Path
import hashlib,json,os,re,tempfile,time

MAX_MD_BYTES=128*1024
ID_RE=re.compile(r"^[a-f0-9]{64}$")
NAME_RE=re.compile(r"^[\w .()\-\u0600-\u06ff]{1,120}\.md$",re.IGNORECASE)

class InvalidMarkdown(ValueError):pass

def validate(filename,content):
    if not isinstance(filename,str) or filename!=Path(filename).name or not NAME_RE.fullmatch(filename):
        raise InvalidMarkdown("Valid .md basename required")
    if not isinstance(content,str) or not content.strip():
        raise InvalidMarkdown("Markdown content required")
    raw=content.encode("utf-8")
    if len(raw)>MAX_MD_BYTES:raise InvalidMarkdown("Markdown exceeds 128 KiB")
    if any(c in content for c in ("\x00","\x1b","\r")):
        raise InvalidMarkdown("Invalid control character")
    if len(content.splitlines())>6000:raise InvalidMarkdown("Too many lines")
    return raw

def headings(content):
    return [ln.strip().lstrip("#").strip()[:160] for ln in content.splitlines()
            if ln.lstrip().startswith("# ") or ln.lstrip().startswith("## ")][:24]

def store(root,filename,content):
    raw=validate(filename,content)
    base=Path(root)/"md_plans"
    base.mkdir(mode=0o700,parents=True,exist_ok=True)
    os.chmod(base,0o700)
    digest=hashlib.sha256(raw).hexdigest()
    text_path=base/(digest+".md")
    meta_path=base/(digest+".json")
    metadata={"id":digest,"filename":filename,"sha256":digest,
        "bytes":len(raw),"created_at":time.time(),"status":"pending_owner_review",
        "headings":headings(content),"training_enabled":False,
        "instructions_active":False,"dataset_imported":False,
        "source":"owner_uploaded_markdown"}
    # Refuse to overwrite an existing document, including same-hash reuploads.
    if text_path.exists():
        if hashlib.sha256(text_path.read_bytes()).hexdigest()!=digest:raise RuntimeError("Stored upload checksum mismatch")
        if not meta_path.exists():raise RuntimeError("Stored upload missing metadata")
        metadata=json.loads(meta_path.read_text())
        return {"document":metadata,"already_exists":True}
    with tempfile.NamedTemporaryFile(dir=base,delete=False,prefix=".md-") as f:
        temp=Path(f.name)
        try:
            os.fchmod(f.fileno(),0o600)
            f.write(raw);f.flush();os.fsync(f.fileno())
        finally:pass
    try:
        # Store uniquely with O_EXCL; do not replace an existing path.
        try:os.link(temp,text_path)
        except FileExistsError:pass
        os.chmod(text_path,0o600)
    finally:temp.unlink(missing_ok=True)
    try:
        fd=os.open(meta_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"w",encoding="utf8") as out:
            json.dump(metadata,out,ensure_ascii=False,indent=2)
            out.flush();os.fsync(out.fileno())
    except FileExistsError:
        metadata=json.loads(meta_path.read_text())
    return {"document":metadata,"already_exists":False}

def list_uploads(root):
    base=Path(root)/"md_plans"
    if not base.is_dir():return []
    result=[]
    for p in base.glob("*.json"):
        if not ID_RE.fullmatch(p.stem) or p.is_symlink():continue
        try:
            item=json.loads(p.read_text())
            if item.get("id")!=p.stem:continue
            result.append(item)
        except (OSError,ValueError,TypeError):continue
    result.sort(key=lambda row:row.get("created_at",0),reverse=True)
    return result

def fetch(root,digest):
    if not ID_RE.fullmatch(digest):raise InvalidMarkdown("Invalid document id")
    path=Path(root)/"md_plans"/(digest+".md")
    meta=Path(root)/"md_plans"/(digest+".json")
    if not path.is_file() or not meta.is_file() or path.is_symlink() or meta.is_symlink():
        raise FileNotFoundError(digest)
    raw=path.read_bytes()
    if len(raw)>MAX_MD_BYTES or hashlib.sha256(raw).hexdigest()!=digest:
        raise RuntimeError("Stored upload checksum mismatch")
    value=json.loads(meta.read_text())
    if value.get("id")!=digest:raise RuntimeError("Stored metadata mismatch")
    return {"document":value,"content":raw.decode("utf8")}
