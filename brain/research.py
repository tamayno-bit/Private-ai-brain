"""Read-only, allowlisted public research adapters.

Never fetches user-supplied URLs; never downloads packages, runs code,
collects user prompts, or imports results into training datasets.
"""
import json,re,time,urllib.parse,urllib.request,urllib.error
from html import unescape

USER_AGENT="PrivateAIBrainResearch/0.1 (owner-operated research prototype)"
ALLOWED={"wikipedia","pypi","crossref","github"}
QUERY_RE=re.compile(r"^[^\x00-\x1f\x7f]{2,120}$")
PACKAGE_RE=re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,90}$")

class ResearchError(ValueError):pass

def validate_query(query):
    if not isinstance(query,str) or not QUERY_RE.fullmatch(query) or query!=query.strip():raise ResearchError("Invalid research query")
    q=query.strip()
    if any(x in q.lower() for x in ("bearer ","api_key=","password=","token=","-----begin","authorization:")):
        raise ResearchError("Potential credential in research query")
    return q

def _json(url,host):
    parsed=urllib.parse.urlsplit(url)
    if parsed.scheme!="https" or parsed.hostname!=host or parsed.username or parsed.password or parsed.port not in (None,443):
        raise ResearchError("Host not allowlisted")
    req=urllib.request.Request(url,headers={"User-Agent":USER_AGENT,"Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=8) as res:
        if res.status!=200 or 'json' not in res.headers.get("Content-Type",""):
            raise ResearchError("Unexpected research source response")
        limit=1200000 if host=="pypi.org" else 220000
        data=res.read(limit+1)
        if len(data)>limit:raise ResearchError("Research source response too large")
        return json.loads(data)

def _strip_html(value):
    return re.sub(r"\s+"," ",unescape(re.sub(r"<[^>]*>"," ",str(value)))).strip()[:480]

def wikipedia(query):
    args=urllib.parse.urlencode({"action":"query","list":"search","srsearch":query,"srlimit":4,"format":"json","utf8":1})
    data=_json("https://en.wikipedia.org/w/api.php?"+args,"en.wikipedia.org")
    items=[]
    for row in data.get("query",{}).get("search",[])[:4]:
        title=str(row.get("title",""))[:170]
        if not title:continue
        items.append({"title":title,"url":"https://en.wikipedia.org/wiki/"+urllib.parse.quote(title.replace(" ","_")),
            "summary":_strip_html(row.get("snippet","")),"source":"Wikipedia",
            "training_rights_verified":False,"role":"reference_only"})
    return items

def pypi(query):
    if not PACKAGE_RE.fullmatch(query):raise ResearchError("PyPI search requires an exact package name")
    data=_json("https://pypi.org/pypi/"+urllib.parse.quote(query,safe="")+"/json","pypi.org")
    info=data.get("info",{})
    name=str(info.get("name") or query)
    version=str(info.get("version") or "")
    description=_strip_html(info.get("summary") or "")
    return [{"title":name+" "+version,"url":"https://pypi.org/project/"+urllib.parse.quote(name,safe="")+"/",
        "summary":description,"source":"PyPI","license_declared":str(info.get("license") or "")[:150],
        "version":version,"training_rights_verified":False,"role":"metadata_reference_only",
        "code_executed":False,"package_installed":False}]

def crossref(query):
    args=urllib.parse.urlencode({"query":query,"rows":4,"select":"DOI,title,published,author,URL"})
    data=_json("https://api.crossref.org/works?"+args,"api.crossref.org")
    rows=[]
    for item in data.get("message",{}).get("items",[])[:4]:
        doi=str(item.get("DOI") or "")
        if not doi.startswith("10.") or any(c.isspace() for c in doi):continue
        title=str((item.get("title") or ["Untitled"])[0])[:200]
        rows.append({"title":title,"url":"https://doi.org/"+urllib.parse.quote(doi,safe="/._-"),
           "summary":"Research paper metadata (Crossref). Verify full text and usage rights separately.",
           "source":"Crossref","doi":doi,"training_rights_verified":False,"role":"metadata_reference_only"})
    return rows

def github(query):
    args=urllib.parse.urlencode({"q":query,"per_page":4})
    data=_json("https://api.github.com/search/repositories?"+args,"api.github.com")
    items=[]
    for row in data.get("items",[])[:4]:
        url=str(row.get("html_url") or "")
        if not url.startswith("https://github.com/"):
            continue
        license_info=row.get("license") or {}
        items.append({"title":str(row.get("full_name") or "")[:160],
            "url":url,"summary":_strip_html(row.get("description") or ""),
            "source":"GitHub public repository metadata",
            "stars":int(row.get("stargazers_count") or 0),
            "language":str(row.get("language") or "")[:60],
            "license_declared":str(license_info.get("spdx_id") or "")[:80],
            "training_rights_verified":False,"role":"metadata_reference_only",
            "repository_cloned":False,"code_executed":False})
    return items

def search(query,provider="wikipedia"):
    query=validate_query(query)
    if provider not in ALLOWED:raise ResearchError("Provider not allowlisted")
    fn={"wikipedia":wikipedia,"pypi":pypi,"crossref":crossref,"github":github}[provider]
    try:results=fn(query)
    except urllib.error.HTTPError as e:
        if e.code==404:return {"query":query,"provider":provider,"results":[],"retrieved_at":time.time(),"status":"not_found"}
        raise ResearchError("Research source HTTP failure") from e
    except (urllib.error.URLError,TimeoutError,OSError,ValueError,KeyError,TypeError) as e:
        raise ResearchError("Research source unavailable or malformed") from e
    return {"query":query,"provider":provider,"results":results,"retrieved_at":time.time(),
        "status":"ok","external_data_for_training":False,"automatic_installation":False}
