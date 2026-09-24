import os, time, uuid, threading, tempfile, traceback, base64
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response

APP_PASSCODE = os.environ.get("APP_PASSCODE", "")
ASSEMBLYAI_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
ELEVENLABS_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

app = FastAPI()
JOBS = {}
LOCK = threading.Lock()


def set_job(jid, **kw):
    with LOCK:
        JOBS.setdefault(jid, {}).update(kw)


def get_job(jid):
    with LOCK:
        return dict(JOBS.get(jid, {}))


def merge(segments, gap=2.0):
    """연속된 같은 화자 구간을 한 문단으로 합친다."""
    out = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        spk = s.get("speaker") or "UNKNOWN"
        if out and out[-1]["speaker"] == spk and s["start"] - out[-1]["end"] < gap:
            out[-1]["text"] += " " + text
            out[-1]["end"] = s["end"]
        else:
            out.append({"speaker": spk, "start": s["start"], "end": s["end"], "text": text})
    return out


# ---------- AssemblyAI ----------
def run_assemblyai(path, language, speakers):
    if not ASSEMBLYAI_KEY:
        raise RuntimeError("ASSEMBLYAI_API_KEY 가 설정되어 있지 않습니다.")
    h = {"authorization": ASSEMBLYAI_KEY}
    with open(path, "rb") as f:
        up = requests.post("https://api.assemblyai.com/v2/upload", headers=h, data=f, timeout=1800)
    up.raise_for_status()
    audio_url = up.json()["upload_url"]

    body = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "language_code": language,
    }
    if speakers:
        body["speakers_expected"] = int(speakers)
    cr = requests.post("https://api.assemblyai.com/v2/transcript", headers=h, json=body, timeout=120)
    cr.raise_for_status()
    tid = cr.json()["id"]

    while True:
        time.sleep(5)
        r = requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}", headers=h, timeout=60)
        r.raise_for_status()
        d = r.json()
        if d["status"] == "completed":
            segs = [
                {
                    "speaker": "SPEAKER_" + str(u.get("speaker") or "?"),
                    "start": u["start"] / 1000.0,
                    "end": u["end"] / 1000.0,
                    "text": u["text"],
                }
                for u in (d.get("utterances") or [])
            ]
            if not segs and d.get("text"):
                segs = [{"speaker": "SPEAKER_A", "start": 0.0, "end": 0.0, "text": d["text"]}]
            return segs
        if d["status"] == "error":
            raise RuntimeError("AssemblyAI: " + str(d.get("error")))


# ---------- ElevenLabs Scribe ----------
def run_elevenlabs(path, language, speakers):
    if not ELEVENLABS_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY 가 설정되어 있지 않습니다.")
    data = {"model_id": "scribe_v1", "diarize": "true", "language_code": language}
    if speakers:
        data["num_speakers"] = str(int(speakers))
    with open(path, "rb") as f:
        r = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": ELEVENLABS_KEY},
            data=data,
            files={"file": (os.path.basename(path), f)},
            timeout=3600,
        )
    r.raise_for_status()
    d = r.json()
    segs, cur = [], None
    for w in d.get("words", []):
        if w.get("type") not in (None, "word", "spacing"):
            continue
        spk = w.get("speaker_id") or "SPEAKER_?"
        if w.get("type") == "spacing":
            if cur:
                cur["text"] += w.get("text", " ")
            continue
        if cur and cur["speaker"] == spk:
            cur["text"] += w.get("text", "")
            cur["end"] = w.get("end", cur["end"])
        else:
            if cur:
                segs.append(cur)
            cur = {"speaker": spk, "start": w.get("start", 0.0), "end": w.get("end", 0.0), "text": w.get("text", "")}
    if cur:
        segs.append(cur)
    if not segs and d.get("text"):
        segs = [{"speaker": "SPEAKER_A", "start": 0.0, "end": 0.0, "text": d["text"]}]
    return segs


RUNNERS = {"assemblyai": run_assemblyai, "elevenlabs": run_elevenlabs}


def worker(jid, path, provider, language, speakers):
    try:
        set_job(jid, status="running", stage="변환하는 중")
        segs = RUNNERS[provider](path, language, speakers)
        blocks = merge(segs)
        set_job(jid, status="done", stage="완료", blocks=blocks,
                speakers=sorted({b["speaker"] for b in blocks}))
    except Exception as e:
        traceback.print_exc()
        msg = str(e)
        if hasattr(e, "response") and getattr(e, "response", None) is not None:
            msg += " / " + e.response.text[:400]
        set_job(jid, status="error", stage="오류", error=msg)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.get("/api/config")
def config():
    return {
        "needs_passcode": bool(APP_PASSCODE),
        "providers": [p for p, k in (("assemblyai", ASSEMBLYAI_KEY), ("elevenlabs", ELEVENLABS_KEY)) if k],
    }


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    passcode: str = Form(""),
    provider: str = Form("assemblyai"),
    language: str = Form("ko"),
    speakers: str = Form(""),
):
    if APP_PASSCODE and passcode != APP_PASSCODE:
        raise HTTPException(status_code=401, detail="암호가 맞지 않습니다.")
    if provider not in RUNNERS:
        raise HTTPException(status_code=400, detail="알 수 없는 변환 업체입니다.")

    suffix = os.path.splitext(file.filename or "audio.m4a")[1] or ".m4a"
    fd, path = tempfile.mkstemp(suffix=suffix, dir="/tmp")
    size = 0
    with os.fdopen(fd, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            out.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="파일이 비어 있습니다.")

    jid = uuid.uuid4().hex[:12]
    set_job(jid, status="queued", stage="올리는 중", name=file.filename, bytes=size, created=time.time())
    threading.Thread(
        target=worker, args=(jid, path, provider, language, speakers.strip()), daemon=True
    ).start()
    return {"job": jid}


@app.get("/api/job/{jid}")
def job(jid: str):
    j = get_job(jid)
    if not j:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return JSONResponse(j)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/favicon.ico")
@app.get("/icon.png")
def icon():
    return Response(base64.b64decode(ICON_B64), media_type="image/png")


@app.get("/manifest.webmanifest")
def manifest():
    return Response(MANIFEST, media_type="application/manifest+json")


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>받아쓰기</title>
<link rel="manifest" href="/manifest.webmanifest">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="받아쓰기">
<link rel="apple-touch-icon" href="/icon.png">
<style>
  :root{
    --ink:#111111; --paper:#ffffff; --card:#ebebeb; --line:#e2e2e2;
    --muted:#8a8a8a; --radius:26px;
  }
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
  html,body{margin:0;padding:0;background:var(--paper);color:var(--ink)}
  body{
    font-family:-apple-system,BlinkMacSystemFont,"Pretendard","Apple SD Gothic Neo","Noto Sans KR",sans-serif;
    padding:8px 16px calc(130px + env(safe-area-inset-bottom));
    max-width:640px;margin:0 auto;-webkit-font-smoothing:antialiased;
  }
  header{display:flex;align-items:center;justify-content:space-between;padding:14px 4px 4px}
  .wordmark{font-size:34px;font-weight:800;letter-spacing:-1.5px}
  .iconbtn{border:0;background:none;font-size:22px;color:var(--ink);padding:6px;cursor:pointer}
  .art{display:block;width:100%;max-width:330px;margin:10px auto 22px}
  .card{background:var(--card);border-radius:var(--radius);padding:22px 22px 18px;margin-bottom:16px}
  .card .label{font-size:15px;color:#4a4a4a;margin-bottom:6px}
  .big{font-size:46px;font-weight:800;letter-spacing:-2px;line-height:1.1;word-break:break-all}
  .sub{font-size:15px;color:#5c5c5c;margin-top:6px}
  .cta{display:flex;align-items:center;justify-content:flex-end;gap:12px;margin-top:20px;
       font-size:20px;font-weight:800;cursor:pointer;background:none;border:0;color:var(--ink);width:100%;padding:0}
  .cta:disabled{color:#a5a5a5;cursor:default}
  .chip{display:inline-flex;align-items:center;gap:6px;background:#fff;border-radius:999px;
        padding:8px 14px;font-size:14px;font-weight:600;border:0}
  .rowbtn{display:flex;align-items:center;gap:14px;width:100%;background:#fff;border:1.5px solid var(--line);
          border-radius:22px;padding:16px 18px;margin-bottom:10px;font-size:17px;color:var(--ink);text-align:left;cursor:pointer}
  .rowbtn .g{width:44px;height:44px;border-radius:14px;background:#f2f2f2;display:grid;place-items:center;font-size:20px;flex:none}
  .rowbtn .t{flex:1;min-width:0}
  .rowbtn .t b{display:block;font-weight:700}
  .rowbtn .t span{font-size:14px;color:var(--muted)}
  h2{font-size:24px;font-weight:800;letter-spacing:-0.8px;margin:26px 4px 12px;display:flex;justify-content:space-between;align-items:baseline}
  h2 em{font-style:normal;font-size:14px;font-weight:500;color:var(--muted)}
  input,select{font:inherit;color:var(--ink);background:#fff;border:1.5px solid var(--line);
               border-radius:16px;padding:12px 14px;width:100%}
  input:focus,select:focus{outline:2px solid #111;outline-offset:-2px}
  .field{margin-bottom:12px}
  .field label{display:block;font-size:13px;color:var(--muted);margin:0 4px 6px}
  .two{display:flex;gap:10px}
  .namegrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
  .namegrid .field{margin:0}
  .namegrid label{display:flex;justify-content:space-between;gap:6px}
  .namegrid label i{font-style:normal;color:#b0b0b0}
  .two>*{flex:1}
  .bar{height:10px;background:#fff;border-radius:99px;overflow:hidden;margin-top:16px}
  .bar i{display:block;height:100%;background:var(--ink);width:0%;transition:width .4s}
  .block{border-bottom:1px solid var(--line);padding:16px 4px}
  .block .meta{display:flex;gap:10px;align-items:baseline;font-size:13px;color:var(--muted);margin-bottom:6px}
  .block .who{font-weight:800;color:var(--ink);font-size:15px}
  .block p{margin:0;font-size:16px;line-height:1.75;white-space:pre-wrap}
  .tabs{position:fixed;left:0;right:0;bottom:0;padding:8px 16px calc(10px + env(safe-area-inset-bottom));
        background:linear-gradient(to top,#fff 70%,rgba(255,255,255,0));}
  .tabs .in{display:flex;background:#fff;border:1.5px solid var(--line);border-radius:999px;padding:6px;max-width:640px;margin:0 auto}
  .tabs button{flex:1;border:0;background:none;font:inherit;font-size:16px;font-weight:700;color:var(--muted);
               padding:12px 0;border-radius:999px;cursor:pointer}
  .tabs button[aria-selected="true"]{background:#f0f0f0;color:var(--ink)}
  .hide{display:none}
  .err{background:#111;color:#fff;border-radius:18px;padding:14px 16px;font-size:14px;line-height:1.6;margin-bottom:14px;white-space:pre-wrap}
  .note{font-size:13px;color:var(--muted);line-height:1.7;margin:4px 4px 20px}
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){--ink:#f2f2f2;--paper:#0f0f0f;--card:#1e1e1e;--line:#2e2e2e;--muted:#9a9a9a}
    :root:not([data-theme="light"]) .chip,
    :root:not([data-theme="light"]) .rowbtn,
    :root:not([data-theme="light"]) input,
    :root:not([data-theme="light"]) select,
    :root:not([data-theme="light"]) .bar,
    :root:not([data-theme="light"]) .tabs .in{background:#171717}
    :root:not([data-theme="light"]) .rowbtn .g{background:#242424}
    :root:not([data-theme="light"]) .tabs{background:linear-gradient(to top,#0f0f0f 70%,rgba(15,15,15,0))}
    :root:not([data-theme="light"]) .tabs button[aria-selected="true"]{background:#282828}
    :root:not([data-theme="light"]) .err{background:#f2f2f2;color:#111}
  }
</style>
</head>
<body>

<header>
  <div class="wordmark">받아쓰기</div>
  <button class="iconbtn" id="gear" aria-label="설정">⚙︎</button>
</header>

<!-- 홈 -->
<section id="view-home">
  <svg class="art" viewBox="0 0 320 210" fill="none" stroke="currentColor" stroke-width="3.4"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <rect x="128" y="26" width="64" height="104" rx="32"/>
    <path d="M104 108c0 31 25 56 56 56s56-25 56-56"/>
    <path d="M160 164v26"/><path d="M132 190h56"/>
    <path d="M232 58c10 14 10 36 0 50"/>
    <path d="M252 44c16 22 16 60 0 82"/>
    <path d="M88 58c-10 14-10 36 0 50"/>
    <path d="M68 44c-16 22-16 60 0 82"/>
    <path d="M146 62h28"/><path d="M146 82h28"/><path d="M146 102h28"/>
  </svg>

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px">
      <div class="label">올릴 녹음</div>
      <button class="chip" id="pick-chip">＋ 파일 고르기</button>
    </div>
    <div class="big" id="fname">비어 있음</div>
    <div class="sub" id="fmeta">최대 90분, m4a·mp3·wav·mp4</div>
    <div class="bar hide" id="bar"><i></i></div>
    <button class="cta" id="go" disabled>변환 시작 <span>→</span></button>
  </div>

  <input type="file" id="file" accept="audio/*,video/mp4" class="hide">

  <div id="err-box"></div>

  <h2>최근 변환 <em id="count">0건</em></h2>
  <div id="recent"></div>
  <p class="note">변환한 글은 이 기기에만 잠시 남습니다. 중요한 내용은 내려받아 두세요.</p>
</section>

<!-- 결과 -->
<section id="view-result" class="hide">
  <div class="card">
    <div class="label" id="r-name">결과</div>
    <div class="big" id="r-count">0문단</div>
    <div class="sub" id="r-speakers">화자 0명</div>
    <div class="cta" style="gap:22px">
      <button class="chip" id="copy">복사</button>
      <button class="chip" id="dl">txt 내려받기</button>
    </div>
  </div>
  <h2>화자 이름 <em id="names-count">0명</em></h2>
  <div id="names"></div>
  <h2>녹취록</h2>
  <div id="blocks"></div>
</section>

<!-- 설정 -->
<section id="view-set" class="hide">
  <h2>설정</h2>
  <div class="field" id="pass-field">
    <label>암호</label>
    <input type="password" id="passcode" placeholder="사이트 암호" autocomplete="off">
  </div>
  <div class="field">
    <label>변환 업체</label>
    <select id="provider"></select>
  </div>
  <div class="two">
    <div class="field">
      <label>언어</label>
      <select id="language">
        <option value="ko" selected>한국어</option>
        <option value="en">영어</option>
      </select>
    </div>
    <div class="field">
      <label>말한 사람 수</label>
      <input id="speakers" inputmode="numeric" placeholder="모르면 비워두기">
    </div>
  </div>
  <p class="note">사람 수를 알면 적어 두는 편이 화자 구분이 정확합니다. 설정은 이 기기에 저장됩니다.</p>
</section>

<nav class="tabs">
  <div class="in">
    <button data-v="home" aria-selected="true">홈</button>
    <button data-v="result" aria-selected="false">결과</button>
    <button data-v="set" aria-selected="false">설정</button>
  </div>
</nav>

<script>
const $ = s => document.querySelector(s);
const store = {
  get(k, d){ try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch(e){ return d; } },
  set(k, v){ try { localStorage.setItem(k, JSON.stringify(v)); } catch(e){} }
};
let picked = null, result = null, names = {}, recent = store.get('recent', []);

function view(v){
  ['home','result','set'].forEach(n => $('#view-'+n).classList.toggle('hide', n !== v));
  document.querySelectorAll('.tabs button').forEach(b => b.setAttribute('aria-selected', b.dataset.v === v));
  window.scrollTo(0,0);
}
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => view(b.dataset.v));
$('#gear').onclick = () => view('set');

['passcode','language','speakers'].forEach(id => {
  const el = $('#'+id);
  el.value = store.get('cfg_'+id, el.value || '');
  el.oninput = () => store.set('cfg_'+id, el.value);
});

fetch('/api/config').then(r => r.json()).then(c => {
  $('#pass-field').classList.toggle('hide', !c.needs_passcode);
  const names = { assemblyai:'AssemblyAI', elevenlabs:'ElevenLabs Scribe' };
  const sel = $('#provider');
  sel.innerHTML = (c.providers.length ? c.providers : ['assemblyai'])
    .map(p => `<option value="${p}">${names[p] || p}</option>`).join('');
  sel.value = store.get('cfg_provider', sel.value);
  sel.onchange = () => store.set('cfg_provider', sel.value);
  if (!c.providers.length) showErr('변환 업체 키가 아직 설정되지 않았습니다. Space 설정에서 API 키를 넣어 주세요.');
});

function showErr(m){ $('#err-box').innerHTML = `<div class="err">${m}</div>`; }
function clearErr(){ $('#err-box').innerHTML = ''; }
function ts(s){
  s = Math.max(0, Math.floor(s));
  const h = String(Math.floor(s/3600)).padStart(2,'0'),
        m = String(Math.floor(s%3600/60)).padStart(2,'0'),
        x = String(s%60).padStart(2,'0');
  return `${h}:${m}:${x}`;
}

$('#pick-chip').onclick = () => $('#file').click();
$('#file').onchange = e => {
  picked = e.target.files[0] || null;
  clearErr();
  $('#fname').textContent = picked ? picked.name : '비어 있음';
  $('#fmeta').textContent = picked
    ? (picked.size/1048576).toFixed(1) + 'MB'
    : '최대 90분, m4a·mp3·wav·mp4';
  $('#go').disabled = !picked;
};

$('#go').onclick = () => {
  if (!picked) return;
  clearErr();
  const fd = new FormData();
  fd.append('file', picked);
  fd.append('passcode', $('#passcode').value);
  fd.append('provider', $('#provider').value);
  fd.append('language', $('#language').value);
  fd.append('speakers', $('#speakers').value);

  $('#go').disabled = true;
  $('#bar').classList.remove('hide');
  const fill = $('#bar i');
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/transcribe');
  xhr.upload.onprogress = e => {
    if (e.lengthComputable) {
      fill.style.width = (e.loaded/e.total*40) + '%';
      $('#fmeta').textContent = '올리는 중 ' + Math.round(e.loaded/e.total*100) + '%';
    }
  };
  xhr.onload = () => {
    if (xhr.status >= 300) {
      let d = {}; try { d = JSON.parse(xhr.responseText); } catch(e){}
      fail(d.detail || ('오류 ' + xhr.status));
      return;
    }
    poll(JSON.parse(xhr.responseText).job, fill);
  };
  xhr.onerror = () => fail('연결이 끊겼습니다. 다시 시도해 주세요.');
  xhr.send(fd);
};

function fail(m){
  showErr(m);
  $('#go').disabled = false;
  $('#bar').classList.add('hide');
  $('#bar i').style.width = '0%';
  $('#fmeta').textContent = picked ? (picked.size/1048576).toFixed(1) + 'MB' : '';
}

function poll(job, fill){
  let n = 0;
  const tick = () => {
    fetch('/api/job/' + job).then(r => r.json()).then(j => {
      n++;
      fill.style.width = Math.min(96, 40 + n*1.2) + '%';
      $('#fmeta').textContent = (j.stage || '처리 중') + ' · ' + Math.floor(n*4/60) + '분 경과';
      if (j.status === 'done'){
        fill.style.width = '100%';
        show(j, picked ? picked.name : '');
      } else if (j.status === 'error'){
        fail(j.error || '변환에 실패했습니다.');
      } else {
        setTimeout(tick, 4000);
      }
    }).catch(() => setTimeout(tick, 6000));
  };
  setTimeout(tick, 3000);
}

function show(j, filename){
  result = j; names = {};
  $('#go').disabled = false;
  $('#bar').classList.add('hide');
  $('#r-name').textContent = filename || '결과';
  $('#r-count').textContent = j.blocks.length + '문단';
  $('#r-speakers').textContent = '화자 ' + (j.speakers || []).length + '명';
  drawNames(j);
  recent.unshift({ name: filename, at: Date.now(), blocks: j.blocks, speakers: j.speakers });
  recent = recent.slice(0, 5);
  store.set('recent', recent);
  drawRecent();
  render();
  view('result');
}

function drawNames(j){
  const list = j.speakers || [];
  const stat = {};
  (j.blocks || []).forEach(b => {
    const t = stat[b.speaker] || (stat[b.speaker] = {n:0, sec:0, first:b.start});
    t.n++; t.sec += Math.max(0, b.end - b.start);
  });
  const order = list.slice().sort((a,c) => (stat[c]?.n || 0) - (stat[a]?.n || 0));
  $('#names-count').textContent = list.length + '명';
  $('#names').className = list.length > 2 ? 'namegrid' : '';
  $('#names').innerHTML = order.map(s => {
    const t = stat[s] || {n:0, sec:0, first:0};
    return `<div class="field">
      <label>${s} <i>${t.n}문단 · ${Math.round(t.sec/60)}분</i></label>
      <input data-spk="${s}" placeholder="이름"></div>`;
  }).join('');
  document.querySelectorAll('[data-spk]').forEach(i => i.oninput = () => {
    names[i.dataset.spk] = i.value.trim(); render();
  });
}
function who(s){ return names[s] || s; }
function render(){
  if (!result) return;
  $('#blocks').innerHTML = result.blocks.map(b => `
    <div class="block">
      <div class="meta"><span class="who">${who(b.speaker)}</span><span>${ts(b.start)}</span></div>
      <p>${b.text.replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}</p>
    </div>`).join('');
}
function asText(){
  return result.blocks.map(b => `[${ts(b.start)}] ${who(b.speaker)}: ${b.text}`).join('\n\n');
}
$('#copy').onclick = async () => {
  try { await navigator.clipboard.writeText(asText()); $('#copy').textContent = '복사됨'; }
  catch(e){ $('#copy').textContent = '복사 실패'; }
  setTimeout(() => $('#copy').textContent = '복사', 1600);
};
$('#dl').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([asText()], {type:'text/plain;charset=utf-8'}));
  a.download = ($('#r-name').textContent || '녹취록').replace(/\.[^.]+$/, '') + '.txt';
  a.click();
};

function drawRecent(){
  $('#count').textContent = recent.length + '건';
  $('#recent').innerHTML = recent.map((r,i) => `
    <button class="rowbtn" data-i="${i}">
      <span class="g">▤</span>
      <span class="t"><b>${r.name || '녹취록'}</b>
      <span>${r.blocks.length}문단 · ${new Date(r.at).toLocaleDateString('ko-KR')}</span></span>
      <span>›</span>
    </button>`).join('') || `<button class="rowbtn" disabled><span class="g">▤</span>
      <span class="t"><b>아직 없습니다</b><span>녹음을 올리면 여기에 쌓입니다</span></span></button>`;
  document.querySelectorAll('#recent .rowbtn[data-i]').forEach(b => b.onclick = () => {
    const r = recent[b.dataset.i];
    result = r; names = {};
    $('#r-name').textContent = r.name || '결과';
    $('#r-count').textContent = r.blocks.length + '문단';
    $('#r-speakers').textContent = '화자 ' + (r.speakers || []).length + '명';
    drawNames(r);
    render(); view('result');
  });
}
drawRecent();
</script>
</body>
</html>
"""

MANIFEST = r"""{
  "name": "받아쓰기",
  "short_name": "받아쓰기",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#ffffff",
  "theme_color": "#ffffff",
  "icons": [
    { "src": "/icon.png", "sizes": "512x512", "type": "image/png", "purpose": "any" },
    { "src": "/icon.png", "sizes": "180x180", "type": "image/png" }
  ]
}
"""

ICON_B64 = "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"
