"""Separate memory: never silently convert personal notes into training samples."""
import sqlite3,datetime
from pathlib import Path
class Memory:
    def __init__(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);self.path=path
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, created TEXT NOT NULL, note TEXT NOT NULL, source TEXT NOT NULL)')
    def add(self,note,source='user'):
        if not isinstance(note,str) or not note.strip() or len(note)>4000:raise ValueError('Invalid note')
        with sqlite3.connect(self.path) as db:
            cur=db.execute('INSERT INTO notes(created,note,source) VALUES(?,?,?)',(datetime.datetime.now(datetime.timezone.utc).isoformat(),note,source));return cur.lastrowid
    def search(self,q,limit=10):
        if not isinstance(q,str) or len(q)>200:raise ValueError('Invalid search')
        with sqlite3.connect(self.path) as db:
            return [{'id':x[0],'note':x[1],'source':x[2]} for x in db.execute('SELECT id,note,source FROM notes WHERE note LIKE ? ORDER BY id DESC LIMIT ?',('%'+q.replace('%','\\%').replace('_','\\_')+'%',min(limit,30))).fetchall()]
