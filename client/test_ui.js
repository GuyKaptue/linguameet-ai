/** @jest-environment jsdom */
'use strict';

// jsdom does not expose TextEncoder/TextDecoder; real browsers do. Borrow Node's for the test environment.
const util = require('util');
if (typeof global.TextEncoder === 'undefined') global.TextEncoder = util.TextEncoder;
if (typeof global.TextDecoder === 'undefined') global.TextDecoder = util.TextDecoder;

const S = require('./subtitles.js');
const App = require('./app.js');

const enc = (obj) => new TextEncoder().encode(typeof obj === 'string' ? obj : JSON.stringify(obj));

function msg(over) {
  return Object.assign(
    {
      type: 'subtitle', version: 1, segment_id: 's1', speaker: 'alice', source_lang: 'en',
      text: 'Hello everyone', translations: { es: 'Hola a todos' }, is_final: true, start_time: 0, end_time: 1.2,
    },
    over
  );
}

const PAGE = `
  <input id="url" value="ws://x"><input id="token" value="tok">
  <button id="connect"></button><button id="disconnect" disabled></button><button id="start-audio" hidden></button><button id="mic" disabled aria-pressed="false">Microphone off</button>
  <select id="speak-lang"><option value="en">en</option><option value="es">es</option></select>
  <select id="listen-lang"><option value="en">en</option><option value="es">es</option><option value="fr">fr</option></select>
  <select id="mode"><option value="both">b</option><option value="translated">t</option><option value="original">o</option></select>
  <p id="status"></p><section id="subtitles"></section><div id="audio"></div>`;

// ---- fake LiveKit client (same surface as livekit-client 2.24.1 that app.js uses) ----
function makeFakeLivekit() {
  const rooms = [];
  class Room {
    constructor() {
      this.handlers = {};
      this.canPlaybackAudio = false;
      this.attrs = [];
      this.connected = false;
      this.mic = []; this.localParticipant = { setAttributes: async (a) => { this.attrs.push(a); }, setMicrophoneEnabled: async (on) => { if (on === 'deny') throw new Error('x'); this.mic.push(on); } };
      rooms.push(this);
    }
    on(ev, fn) { (this.handlers[ev] = this.handlers[ev] || []).push(fn); return this; }
    off(ev, fn) { this.handlers[ev] = (this.handlers[ev] || []).filter((f) => f !== fn); return this; }
    emit(ev, ...args) { (this.handlers[ev] || []).slice().forEach((f) => f(...args)); }
    async connect(url, token) {
      if (token === 'bad') throw new Error('invalid token');
      this.url = url; this.token = token; this.connected = true;
    }
    async disconnect() { this.connected = false; this.emit('disconnected'); }
    async startAudio() { this.canPlaybackAudio = true; this.emit('audioPlaybackChanged', true); }
  }
  return {
    rooms,
    Room,
    RoomEvent: {
      DataReceived: 'dataReceived', TrackSubscribed: 'trackSubscribed', TrackUnsubscribed: 'trackUnsubscribed',
      Disconnected: 'disconnected', AudioPlaybackStatusChanged: 'audioPlaybackChanged',
    },
    Track: { Kind: { Audio: 'audio', Video: 'video' } },
  };
}

function setup(storage) {
  document.body.innerHTML = PAGE;
  const LK = makeFakeLivekit();
  const timers = [];
  const app = App.createApp({
    document, LivekitClient: LK, storage,
    setInterval: (fn) => { timers.push(fn); return timers.length; },
    clearInterval: () => {},
  });
  return { app, LK, timers, $: (id) => document.getElementById(id) };
}

const flush = () => new Promise((r) => setTimeout(r, 0));
const texts = (sel) => Array.from(document.querySelectorAll(sel)).map((e) => e.textContent);

describe('parseSubtitlePayload', () => {
  test('decodes a Uint8Array JSON packet', () => {
    const m = S.parseSubtitlePayload(enc(msg()));
    expect(m.segment_id).toBe('s1');
    expect(m.translations.es).toBe('Hola a todos');
    expect(m.is_final).toBe(true);
  });

  test('decodes non-ASCII text as UTF-8', () => {
    const m = S.parseSubtitlePayload(enc(msg({ text: '你好，世界', translations: { fr: 'Bonjour le monde — é' } })));
    expect(m.text).toBe('你好，世界');
    expect(m.translations.fr).toBe('Bonjour le monde — é');
  });

  test.each([
    ['not json', 'nope{'],
    ['wrong type', JSON.stringify({ type: 'chat' })],
    ['null', 'null'],
    ['missing text', JSON.stringify({ type: 'subtitle', segment_id: 'a', source_lang: 'en' })],
    ['future version', JSON.stringify(msg({ version: 2 }))],
    ['translations array', JSON.stringify(msg({ translations: ['x'] }))],
  ])('rejects %s', (_n, payload) => {
    expect(() => S.parseSubtitlePayload(payload)).toThrow();
  });

  test('drops non-string translation values and defaults optional fields', () => {
    const m = S.parseSubtitlePayload(JSON.stringify({
      type: 'subtitle', segment_id: 'a', source_lang: 'en', text: 'x', translations: { es: 5, fr: 'ok' },
    }));
    expect(m.translations).toEqual({ fr: 'ok' });
    expect(m.speaker).toBe('');
    expect(m.is_final).toBe(false);
  });
});

describe('pickText', () => {
  const m = S.parseSubtitlePayload(JSON.stringify(msg()));
  test('both shows original and translation', () => {
    expect(S.pickText(m, { mode: 'both', lang: 'es' })).toEqual({ original: 'Hello everyone', translated: 'Hola a todos', pending: false });
  });
  test('translated only hides the original', () => {
    expect(S.pickText(m, { mode: 'translated', lang: 'es' })).toEqual({ original: null, translated: 'Hola a todos', pending: false });
  });
  test('original only', () => {
    expect(S.pickText(m, { mode: 'original', lang: 'es' }).translated).toBeNull();
  });
  test('listener language equals source language: no translation needed', () => {
    expect(S.pickText(m, { mode: 'translated', lang: 'en' })).toEqual({ original: 'Hello everyone', translated: null, pending: false });
  });
  test('missing translation marks pending and falls back to the original', () => {
    expect(S.pickText(m, { mode: 'translated', lang: 'fr' })).toEqual({ original: 'Hello everyone', translated: null, pending: true });
  });
});

describe('SubtitleStore', () => {
  const parse = (o) => S.parseSubtitlePayload(JSON.stringify(msg(o)));

  test('partial then final updates the same segment', () => {
    const st = new S.SubtitleStore();
    st.apply(parse({ is_final: false, text: 'Hel' }));
    st.apply(parse({ is_final: true, text: 'Hello' }));
    expect(st.list()).toHaveLength(1);
    expect(st.list()[0].text).toBe('Hello');
  });

  test('late partial after final is ignored', () => {
    const st = new S.SubtitleStore();
    st.apply(parse({ is_final: true, text: 'Hello' }));
    expect(st.apply(parse({ is_final: false, text: 'Hel' }))).toBe(false);
    expect(st.list()[0].text).toBe('Hello');
  });

  test('translations accumulate across updates of one segment', () => {
    const st = new S.SubtitleStore();
    st.apply(parse({ is_final: false, translations: { es: 'Hola' } }));
    st.apply(parse({ is_final: true, translations: { fr: 'Salut' } }));
    expect(st.list()[0].translations).toEqual({ es: 'Hola', fr: 'Salut' });
  });

  test('keeps only maxSegments, dropping the oldest', () => {
    const st = new S.SubtitleStore({ maxSegments: 2 });
    ['a', 'b', 'c'].forEach((id) => st.apply(parse({ segment_id: id })));
    expect(st.list().map((x) => x.segment_id)).toEqual(['b', 'c']);
  });

  test('prune removes only expired final segments', () => {
    let t = 0;
    const st = new S.SubtitleStore({ finalTtlMs: 1000, now: () => t });
    st.apply(parse({ segment_id: 'old', is_final: true }));
    st.apply(parse({ segment_id: 'live', is_final: false }));
    t = 1500;
    expect(st.prune()).toBe(true);
    expect(st.list().map((x) => x.segment_id)).toEqual(['live']);
    expect(st.prune()).toBe(false);
  });
});

describe('SubtitleRenderer', () => {
  test('renders translated text in the DOM', () => {
    document.body.innerHTML = '<section id="c"></section>';
    const r = new S.SubtitleRenderer(document.getElementById('c'), { mode: 'both', lang: 'es' });
    r.render([S.parseSubtitlePayload(JSON.stringify(msg()))]);
    expect(texts('.translated')).toEqual(['Hola a todos']);
    expect(texts('.original')).toEqual(['Hello everyone']);
    expect(document.getElementById('c').getAttribute('aria-live')).toBe('polite');
  });

  test('updates nodes in place and removes stale ones', () => {
    document.body.innerHTML = '<section id="c"></section>';
    const c = document.getElementById('c');
    const r = new S.SubtitleRenderer(c, { mode: 'both', lang: 'es' });
    const mk = (o) => S.parseSubtitlePayload(JSON.stringify(msg(o)));
    r.render([mk({ is_final: false, text: 'Hel', translations: {} })]);
    const node = c.querySelector('.subtitle');
    expect(node.getAttribute('data-final')).toBe('false');
    expect(node.classList.contains('pending')).toBe(true);
    r.render([mk({ is_final: true })]);
    expect(c.querySelector('.subtitle')).toBe(node); // same element reused
    expect(node.getAttribute('data-final')).toBe('true');
    expect(node.classList.contains('pending')).toBe(false);
    r.render([]);
    expect(c.children).toHaveLength(0);
  });

  test('hostile text is rendered as text, never as markup', () => {
    document.body.innerHTML = '<section id="c"></section>';
    const c = document.getElementById('c');
    const r = new S.SubtitleRenderer(c, { mode: 'both', lang: 'es' });
    const evil = '<img src=x onerror="window.__pwned=1"><script>window.__pwned=1</script>';
    r.render([S.parseSubtitlePayload(JSON.stringify(msg({
      text: evil, translations: { es: evil }, speaker: evil, segment_id: 'a"]x',
    })))]);
    expect(c.querySelector('img')).toBeNull();
    expect(c.querySelector('script')).toBeNull();
    expect(c.querySelector('.translated').textContent).toBe(evil);
    expect(window.__pwned).toBeUndefined();
  });

  test('preserves order of segments', () => {
    document.body.innerHTML = '<section id="c"></section>';
    const c = document.getElementById('c');
    const r = new S.SubtitleRenderer(c, { mode: 'original', lang: '' });
    const mk = (id) => S.parseSubtitlePayload(JSON.stringify(msg({ segment_id: id, text: id })));
    r.render([mk('a'), mk('b'), mk('c')]);
    r.render([mk('b'), mk('c'), mk('d')]);
    expect(texts('.original')).toEqual(['b', 'c', 'd']);
  });
});

describe('attachDataChannel', () => {
  test('forwards subtitle-topic packets, ignores other topics, reports malformed ones, unsubscribes', () => {
    const LK = makeFakeLivekit();
    const room = new LK.Room();
    const got = []; const errs = [];
    const off = S.attachDataChannel(room, LK.RoomEvent, (m) => got.push(m), (e) => errs.push(e));
    room.emit('dataReceived', enc(msg()), null, 0, 'subtitles');
    room.emit('dataReceived', enc(msg({ segment_id: 'zz' })), null, 0, 'chat');
    room.emit('dataReceived', enc('garbage'), null, 0, 'subtitles');
    expect(got.map((m) => m.segment_id)).toEqual(['s1']);
    expect(errs).toHaveLength(1);
    off();
    room.emit('dataReceived', enc(msg()), null, 0, 'subtitles');
    expect(got).toHaveLength(1);
  });
});

describe('app wiring', () => {
  test('receiving a translation over the data channel shows it on screen', async () => {
    const { app, LK, $ } = setup(null);
    $('listen-lang').value = 'es';
    $('listen-lang').dispatchEvent(new Event('change'));
    await app.connect();
    const room = LK.rooms[0];
    room.emit('dataReceived', enc(msg()), { identity: 'gateway' }, 0, 'subtitles');
    expect(texts('.subtitle .translated')).toEqual(['Hola a todos']);
    expect($('status').textContent).toBe('Connected.');
  });

  test('connect announces language attributes; changing preferences re-announces', async () => {
    const { app, LK, $ } = setup(null);
    $('speak-lang').value = 'es';
    $('listen-lang').value = 'fr';
    await app.connect();
    const room = LK.rooms[0];
    expect(room.attrs[0]).toEqual({ listen_lang: 'fr', speak_lang: 'es' });
    $('listen-lang').value = 'en';
    $('listen-lang').dispatchEvent(new Event('change'));
    await flush();
    expect(room.attrs[1]).toEqual({ listen_lang: 'en', speak_lang: 'es' });
  });

  test('switching display mode re-renders existing subtitles', async () => {
    const { app, LK, $ } = setup(null);
    $('listen-lang').value = 'es';
    $('listen-lang').dispatchEvent(new Event('change'));
    await app.connect();
    LK.rooms[0].emit('dataReceived', enc(msg()), null, 0, 'subtitles');
    expect(texts('.original').filter(Boolean)).toEqual(['Hello everyone']);
    $('mode').value = 'translated';
    $('mode').dispatchEvent(new Event('change'));
    expect(document.querySelector('.original').hidden).toBe(true);
    expect(document.querySelector('.translated').hidden).toBe(false);
  });

  test('malformed packet shows a warning and keeps the app alive', async () => {
    const { app, LK, $ } = setup(null);
    await app.connect();
    LK.rooms[0].emit('dataReceived', enc('{bad'), null, 0, 'subtitles');
    expect($('status').getAttribute('data-state')).toBe('warn');
    LK.rooms[0].emit('dataReceived', enc(msg()), null, 0, 'subtitles');
    expect(document.querySelectorAll('.subtitle')).toHaveLength(1);
  });

  test('connect without a token reports an error and creates no room', async () => {
    const { app, LK, $ } = setup(null);
    $('token').value = '';
    await app.connect();
    expect(LK.rooms).toHaveLength(0);
    expect($('status').getAttribute('data-state')).toBe('error');
  });

  test('failed connection cleans up and re-enables the join button', async () => {
    const { app, $ } = setup(null);
    $('token').value = 'bad';
    await app.connect();
    expect($('status').textContent).toMatch(/invalid token/);
    expect($('connect').disabled).toBe(false);
    expect(app.state.room).toBeNull();
  });

  test('subscribed audio track is attached; video is ignored; unsubscribe removes it', async () => {
    const { app, LK, $ } = setup(null);
    await app.connect();
    const room = LK.rooms[0];
    const el = document.createElement('audio');
    const audioTrack = { kind: 'audio', attach: () => el, detach: () => [el] };
    const videoTrack = { kind: 'video', attach: () => { throw new Error('should not attach video'); } };
    room.emit('trackSubscribed', videoTrack);
    room.emit('trackSubscribed', audioTrack);
    expect($('audio').contains(el)).toBe(true);
    room.emit('trackUnsubscribed', audioTrack);
    expect($('audio').contains(el)).toBe(false);
  });

  test('only the TTS track of the listen language is played; switching language swaps tracks', async () => {
    const { app, LK, $ } = setup(null);
    $('listen-lang').value = 'es';
    await app.connect();
    const room = LK.rooms[0];
    const mk = (kind) => {
      const el = document.createElement('audio');
      el.dataset.kind = kind;
      return { kind: 'audio', attach: () => el, detach: () => [el], el };
    };
    const es = mk('es'), fr = mk('fr'), human = mk('human');
    room.emit('trackSubscribed', es, { trackName: 'tts-es' });
    room.emit('trackSubscribed', fr, { trackName: 'tts-fr' });
    room.emit('trackSubscribed', human, { trackName: 'microphone' });
    const playing = () => Array.from($('audio').children).map((e) => e.dataset.kind).sort();
    expect(playing()).toEqual(['es', 'human']);
    $('listen-lang').value = 'fr';
    $('listen-lang').dispatchEvent(new Event('change'));
    expect(playing()).toEqual(['fr', 'human']);
    room.emit('trackUnsubscribed', fr);
    expect(playing()).toEqual(['human']);
  });

  test('microphone button publishes the mic via LiveKit, toggles, and is disabled when not connected', async () => {
    const { app, LK, $ } = setup(null);
    expect($('mic').disabled).toBe(true);
    await app.connect();
    expect($('mic').disabled).toBe(false);
    $('mic').click(); await flush();
    expect(LK.rooms[0].mic).toEqual([true]);
    expect($('mic').textContent).toBe('Microphone on');
    $('mic').click(); await flush();
    expect(LK.rooms[0].mic).toEqual([true, false]);
    expect($('mic').getAttribute('aria-pressed')).toBe('false');
    await app.disconnect();
    expect($('mic').disabled).toBe(true);
  });

  test('microphone permission errors are shown, not thrown', async () => {
    const { app, LK, $ } = setup(null);
    await app.connect();
    LK.rooms[0].localParticipant.setMicrophoneEnabled = async () => { throw new Error('Permission denied'); };
    $('mic').click(); await flush();
    expect($('status').textContent).toMatch(/Permission denied/);
    expect($('mic').getAttribute('aria-pressed')).toBe('false');
  });

  test('enable-audio button appears when playback is blocked and starts audio on click', async () => {
    const { app, LK, $ } = setup(null);
    await app.connect();
    const room = LK.rooms[0];
    expect($('start-audio').hidden).toBe(false);
    $('start-audio').click();
    await flush();
    expect(room.canPlaybackAudio).toBe(true);
    expect($('start-audio').hidden).toBe(true);
  });

  test('leaving clears subtitles and restores buttons; handlers are detached', async () => {
    const { app, LK, $ } = setup(null);
    await app.connect();
    const room = LK.rooms[0];
    room.emit('dataReceived', enc(msg()), null, 0, 'subtitles');
    expect(document.querySelectorAll('.subtitle')).toHaveLength(1);
    await app.disconnect();
    expect(document.querySelectorAll('.subtitle')).toHaveLength(0);
    expect($('connect').disabled).toBe(false);
    expect($('disconnect').disabled).toBe(true);
    expect((room.handlers.dataReceived || []).length).toBe(0);
  });

  test('timer prunes expired finals', async () => {
    const { app, LK, timers } = setup(null);
    await app.connect();
    LK.rooms[0].emit('dataReceived', enc(msg()), null, 0, 'subtitles');
    app.store.finalTtlMs = -1; // everything expired
    timers[0]();
    expect(document.querySelectorAll('.subtitle')).toHaveLength(0);
  });

  test('preferences persist to storage and are restored', () => {
    const mem = {};
    const storage = { getItem: (k) => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = v; } };
    const a = setup(storage);
    a.$('listen-lang').value = 'fr';
    a.$('listen-lang').dispatchEvent(new Event('change'));
    const b = setup(storage);
    expect(b.$('listen-lang').value).toBe('fr');
  });

  test('works when storage throws', () => {
    const storage = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); } };
    const a = setup(storage);
    a.$('listen-lang').value = 'fr';
    expect(() => a.$('listen-lang').dispatchEvent(new Event('change'))).not.toThrow();
  });
});

describe('index.html consistency', () => {
  test('every element id used by app.js exists in index.html', () => {
    const fs = require('fs');
    const html = fs.readFileSync(require('path').join(__dirname, 'index.html'), 'utf8');
    for (const id of ['url', 'token', 'connect', 'disconnect', 'listen-lang', 'speak-lang', 'mode', 'status', 'subtitles', 'audio', 'start-audio', 'mic']) {
      expect(html).toContain('id="' + id + '"');
    }
    expect(html).toContain('livekit-client@2.24.1');
  });
});
