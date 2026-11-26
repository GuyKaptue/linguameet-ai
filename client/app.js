/* Browser glue: connect to a LiveKit room, play translated audio, render subtitles.
 *
 * LiveKit JS API used (livekit-client 2.24.1 type definitions): new Room(), room.connect(url, token),
 * room.disconnect(), room.startAudio(), room.canPlaybackAudio, room.localParticipant.setAttributes(),
 * RoomEvent.{DataReceived, TrackSubscribed, TrackUnsubscribed, Disconnected, AudioPlaybackStatusChanged},
 * Track.Kind.Audio, track.attach() / track.detach().
 *
 * Listener preferences are published as participant attributes so the server knows which
 * translations / TTS voice to produce for this participant (keys must match gateway/protocol.py).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./subtitles.js'));
  else root.LinguaApp = factory(root.LinguaSubtitles);
})(typeof self !== 'undefined' ? self : this, function (Subtitles) {
  'use strict';

  const ATTR_LISTEN_LANG = 'listen_lang';
  const ATTR_SPEAK_LANG = 'speak_lang';
  const TTS_TRACK_PREFIX = 'tts-'; // keep in sync with gateway/protocol.py
  const PREF_KEY = 'linguameet.prefs';

  function loadPrefs(storage) {
    try {
      const raw = storage && storage.getItem(PREF_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (e) {
      return {};
    }
  }

  function savePrefs(storage, prefs) {
    try {
      if (storage) storage.setItem(PREF_KEY, JSON.stringify(prefs));
    } catch (e) {
      /* storage unavailable (private mode): preferences just are not remembered */
    }
  }

  /**
   * createApp({ document, LivekitClient, storage, setInterval, clearInterval })
   * Expects the DOM ids used in index.html.
   */
  function createApp(env) {
    const doc = env.document;
    const LK = env.LivekitClient;
    const $ = (id) => doc.getElementById(id);
    const saved = loadPrefs(env.storage);

    const els = {
      url: $('url'), token: $('token'), connect: $('connect'), disconnect: $('disconnect'),
      listen: $('listen-lang'), speak: $('speak-lang'), mode: $('mode'),
      status: $('status'), mic: $('mic'), subtitles: $('subtitles'), audio: $('audio'), startAudio: $('start-audio'),
    };
    if (saved.listen) els.listen.value = saved.listen;
    if (saved.speak) els.speak.value = saved.speak;
    if (saved.mode) els.mode.value = saved.mode;

    const store = new Subtitles.SubtitleStore({ maxSegments: 6, finalTtlMs: 10000 });
    const renderer = new Subtitles.SubtitleRenderer(els.subtitles, { mode: els.mode.value, lang: els.listen.value });
    const state = { room: null, detach: null, timer: null };
    const audioTracks = new Map(); // track -> { name, el|null }; TTS tracks are named "tts-<lang>"

    function wantsTrack(name) {
      // human microphones (any other name) are always audible; synthesized speech only in the listen language
      return !name || name.indexOf(TTS_TRACK_PREFIX) !== 0 || name === TTS_TRACK_PREFIX + els.listen.value;
    }

    function syncAudio() {
      audioTracks.forEach((entry, track) => {
        const want = wantsTrack(entry.name);
        if (want && !entry.el) {
          entry.el = track.attach();
          els.audio.appendChild(entry.el);
        } else if (!want && entry.el) {
          track.detach().forEach((el) => el.remove());
          entry.el = null;
        }
      });
    }

    function setStatus(text, kind) {
      els.status.textContent = text;
      els.status.setAttribute('data-state', kind || 'idle');
    }

    function redraw() {
      renderer.render(store.list());
    }

    function currentPrefs() {
      return { mode: els.mode.value, lang: els.listen.value, listen: els.listen.value, speak: els.speak.value };
    }

    async function announce() {
      if (!state.room) return;
      const attrs = {};
      attrs[ATTR_LISTEN_LANG] = els.listen.value;
      attrs[ATTR_SPEAK_LANG] = els.speak.value;
      await state.room.localParticipant.setAttributes(attrs);
    }

    function onPrefsChanged() {
      const p = currentPrefs();
      renderer.setPrefs({ mode: p.mode, lang: p.lang });
      savePrefs(env.storage, p);
      redraw();
      syncAudio();
      announce().catch((e) => setStatus('Could not update preferences: ' + e.message, 'error'));
    }
    [els.listen, els.speak, els.mode].forEach((el) => el.addEventListener('change', onPrefsChanged));

    function handleSubtitle(msg) {
      if (store.apply(msg)) redraw();
    }

    async function connect() {
      if (state.room) return;
      const url = els.url.value.trim();
      const token = els.token.value.trim();
      if (!url || !token) {
        setStatus('Enter the server URL and an access token.', 'error');
        return;
      }
      const room = new LK.Room();
      state.room = room;
      state.detach = Subtitles.attachDataChannel(room, LK.RoomEvent, handleSubtitle, (err) =>
        setStatus('Ignored a malformed subtitle: ' + err.message, 'warn')
      );
      room.on(LK.RoomEvent.TrackSubscribed, (track, publication) => {
        if (track.kind !== LK.Track.Kind.Audio) return;
        audioTracks.set(track, { name: (publication && publication.trackName) || '', el: null });
        syncAudio();
      });
      room.on(LK.RoomEvent.TrackUnsubscribed, (track) => {
        track.detach().forEach((el) => el.remove());
        audioTracks.delete(track);
      });
      room.on(LK.RoomEvent.AudioPlaybackStatusChanged, () => {
        els.startAudio.hidden = room.canPlaybackAudio;
      });
      room.on(LK.RoomEvent.Disconnected, () => cleanup('Disconnected.'));
      try {
        setStatus('Connecting…', 'connecting');
        await room.connect(url, token);
        await announce();
        els.connect.disabled = true;
        els.disconnect.disabled = false;
        els.mic.disabled = false;
        els.startAudio.hidden = room.canPlaybackAudio;
        state.timer = (env.setInterval || setInterval)(() => {
          if (store.prune()) redraw();
        }, 1000);
        setStatus('Connected.', 'connected');
      } catch (err) {
        cleanup('Connection failed: ' + err.message, 'error');
      }
    }

    function cleanup(message, kind) {
      if (state.detach) state.detach();
      if (state.timer) (env.clearInterval || clearInterval)(state.timer);
      state.room = null;
      state.detach = null;
      state.timer = null;
      store.clear();
      redraw();
      els.audio.replaceChildren();
      audioTracks.clear();
      els.connect.disabled = false;
      els.disconnect.disabled = true;
      els.mic.disabled = true;
      els.mic.setAttribute('aria-pressed', 'false');
      els.mic.textContent = 'Microphone off';
      els.startAudio.hidden = true;
      setStatus(message, kind || 'idle');
    }

    async function disconnect() {
      if (state.room) await state.room.disconnect();
      else cleanup('Disconnected.');
    }

    async function toggleMic() {
      if (!state.room) return;
      const enable = els.mic.getAttribute('aria-pressed') !== 'true';
      try {
        await state.room.localParticipant.setMicrophoneEnabled(enable);
        els.mic.setAttribute('aria-pressed', String(enable));
        els.mic.textContent = enable ? 'Microphone on' : 'Microphone off';
      } catch (err) {
        setStatus('Microphone unavailable: ' + err.message, 'error');
      }
    }
    els.mic.addEventListener('click', toggleMic);

    els.connect.addEventListener('click', connect);
    els.disconnect.addEventListener('click', disconnect);
    els.startAudio.addEventListener('click', () => state.room && state.room.startAudio());
    setStatus('Not connected.', 'idle');
    redraw();

    return { connect, disconnect, handleSubtitle, store, renderer, state, currentPrefs, ATTR_LISTEN_LANG, ATTR_SPEAK_LANG };
  }

  return { createApp, loadPrefs, savePrefs, ATTR_LISTEN_LANG, ATTR_SPEAK_LANG, PREF_KEY };
});
