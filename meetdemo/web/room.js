/* Demo meeting room page. Subtitle parsing, state and rendering come from the real client module (/lib/subtitles.js). */
(function () {
  'use strict';
  const S = window.LinguaSubtitles;
  const $ = (id) => document.getElementById(id);
  const store = new S.SubtitleStore({ maxSegments: 5, finalTtlMs: 20000 });
  const renderer = new S.SubtitleRenderer($('subtitles'), { mode: 'both', lang: 'en' });
  let ws = null, ctx = null, soundOn = false, nextAt = 0, me = 'guest', participants = [];

  function status(text) { $('status').textContent = text; }
  function redraw() { renderer.render(store.list()); }

  function buildTiles(list) {
    const tiles = $('tiles');
    tiles.replaceChildren();
    list.forEach((p) => {
      const t = document.createElement('div');
      t.className = 'tile';
      t.dataset.id = p.id;
      const who = document.createElement('div'); who.className = 'who';
      const av = document.createElement('div'); av.className = 'avatar'; av.textContent = p.name[0];
      const col = document.createElement('div');
      const name = document.createElement('div'); name.className = 'name'; name.textContent = p.name;
      const meta = document.createElement('div'); meta.className = 'meta'; meta.textContent = 'speaks ' + (window.__LANGS[p.lang] || p.lang) + (p.seconds ? ' · ' + p.seconds + ' s clip' : '');
      col.append(name, meta); who.append(av, col);
      const bars = document.createElement('div'); bars.className = 'bars';
      for (let i = 0; i < 4; i++) bars.appendChild(document.createElement('i'));
      const say = document.createElement('div'); say.className = 'say';
      say.textContent = p.phrases ? 'Script: ' + p.phrases.join(' / ') : 'That is you (listening)';
      const btn = document.createElement('button');
      btn.type = 'button';
      if (p.phrases) { btn.textContent = 'Play ' + p.name + "'s recording"; btn.addEventListener('click', () => send({ op: 'play', who: p.id })); }
      else { btn.textContent = 'Listening'; btn.disabled = true; }
      t.append(who, bars, say, btn);
      tiles.appendChild(t);
    });
  }

  function tile(id) { return document.querySelector('.tile[data-id="' + id + '"]'); }
  function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

  function playAudio(buf) {
    if (!soundOn || !ctx) return;
    const view = new DataView(buf);
    const rate = view.getUint32(0, true);
    const n = (buf.byteLength - 4) >> 1;
    const pcm = new Int16Array(buf.slice(4, 4 + n * 2));
    const ab = ctx.createBuffer(1, n, rate);
    const ch = ab.getChannelData(0);
    for (let i = 0; i < n; i++) ch[i] = pcm[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = ab; src.connect(ctx.destination);
    nextAt = Math.max(nextAt, ctx.currentTime + 0.05);
    src.start(nextAt); nextAt += ab.duration;
  }

  function onMessage(ev) {
    if (typeof ev.data !== 'string') { playAudio(ev.data); return; }
    const m = JSON.parse(ev.data);
    if (m.kind === 'hello') {
      window.__LANGS = m.languages; me = m.you; participants = m.participants;
      const sel = $('listen-lang');
      if (!sel.options.length) Object.entries(m.languages).forEach(([c, n]) => sel.add(new Option(n, c)));
      buildTiles(participants.concat([{ id: 'you', name: 'You (' + me + ')', lang: sel.value }]));
      document.querySelector('.tile[data-id="you"] .meta').textContent = 'listening in ' + m.languages[sel.value];
      send({ op: 'prefs', listen: sel.value });
      status('In the room. Press "Play" on a participant to make them speak.');
    } else if (m.kind === 'subtitle') {
      const msg = S.parseSubtitlePayload(JSON.stringify(m.payload));
      if (store.apply(msg)) redraw();
      if (m.payload.is_final && m.after_gate_ms != null) {
        const el = $('latency');
        el.hidden = false;
        el.textContent = 'Last utterance: subtitle published ' + Math.round(m.after_gate_ms) + ' ms after the VAD gate closed (+ ' + m.gate_hangover_ms + ' ms silence hang-over). Mock models, so this is the pipeline overhead only.';
      }
    } else if (m.kind === 'vad') {
      const t = tile(m.speaker);
      if (t) t.classList.toggle('speaking', m.event === 'start');
    } else if (m.kind === 'playing' && m.event === 'end') {
      status('In the room.');
    }
  }

  function connect() {
    const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
    ws = new WebSocket(proto + location.host + '/ws');
    ws.binaryType = 'arraybuffer';
    ws.onmessage = onMessage;
    ws.onclose = () => status('Disconnected. Reload to rejoin.');
  }

  function prefs() {
    const listen = $('listen-lang').value;
    renderer.setPrefs({ mode: $('mode').value, lang: listen });
    redraw();
    send({ op: 'prefs', listen: listen });
    const you = document.querySelector('.tile[data-id="you"] .meta');
    if (you && window.__LANGS) you.textContent = 'listening in ' + window.__LANGS[listen];
  }
  $('listen-lang').addEventListener('change', prefs);
  $('mode').addEventListener('change', prefs);
  $('sound').addEventListener('click', () => {
    soundOn = !soundOn;
    if (soundOn) { ctx = ctx || new (window.AudioContext || window.webkitAudioContext)(); ctx.resume(); }
    $('sound').textContent = soundOn ? 'Sound on' : 'Sound off';
    $('sound').setAttribute('aria-pressed', String(soundOn));
  });
  window.__LANGS = {};
  connect();
})();
