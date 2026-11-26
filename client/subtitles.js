/* Subtitle protocol, state and DOM rendering for the LinguaMeet web client.
 *
 * Works as a browser global (window.LinguaSubtitles) and as a CommonJS module (Jest).
 * Payload format = shared/types.py SubtitleMessage (JSON, topic "subtitles", version 1).
 * All text is written with textContent, never innerHTML, so payloads cannot inject markup.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.LinguaSubtitles = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const SUBTITLE_TOPIC = 'subtitles';
  const SCHEMA_VERSION = 1;
  const MODES = ['both', 'translated', 'original'];

  /** Decode and validate a data-channel payload (Uint8Array or JSON string). Throws on bad input. */
  function parseSubtitlePayload(payload) {
    const text = typeof payload === 'string' ? payload : new TextDecoder('utf-8').decode(payload);
    let obj;
    try {
      obj = JSON.parse(text);
    } catch (e) {
      throw new Error('subtitle payload is not valid JSON');
    }
    if (!obj || typeof obj !== 'object' || obj.type !== 'subtitle') throw new Error('not a subtitle message');
    if (typeof obj.version === 'number' && obj.version > SCHEMA_VERSION) {
      throw new Error('unsupported subtitle schema version ' + obj.version);
    }
    for (const key of ['segment_id', 'source_lang', 'text']) {
      if (typeof obj[key] !== 'string') throw new Error('subtitle field "' + key + '" must be a string');
    }
    const translations = {};
    if (obj.translations != null) {
      if (typeof obj.translations !== 'object' || Array.isArray(obj.translations)) {
        throw new Error('subtitle field "translations" must be an object');
      }
      for (const [lang, value] of Object.entries(obj.translations)) {
        if (typeof value === 'string') translations[lang] = value;
      }
    }
    return {
      segment_id: obj.segment_id,
      speaker: typeof obj.speaker === 'string' ? obj.speaker : '',
      source_lang: obj.source_lang,
      text: obj.text,
      translations,
      is_final: obj.is_final === true,
      start_time: Number(obj.start_time) || 0,
      end_time: Number(obj.end_time) || 0,
    };
  }

  /** Keeps the most recent segments; partial updates replace each other, late partials are ignored. */
  class SubtitleStore {
    constructor(opts) {
      const o = opts || {};
      this.maxSegments = o.maxSegments || 6;
      this.finalTtlMs = o.finalTtlMs == null ? 8000 : o.finalTtlMs;
      this.now = o.now || Date.now;
      this._segments = new Map(); // segment_id -> { msg, updatedAt, seq }
      this._seq = 0;
    }

    apply(msg) {
      const existing = this._segments.get(msg.segment_id);
      if (existing && existing.msg.is_final && !msg.is_final) return false; // out-of-order partial
      const merged = existing
        ? Object.assign({}, msg, { translations: Object.assign({}, existing.msg.translations, msg.translations) })
        : msg;
      this._segments.set(msg.segment_id, {
        msg: merged,
        updatedAt: this.now(),
        seq: existing ? existing.seq : this._seq++,
      });
      this._enforceLimit();
      return true;
    }

    _enforceLimit() {
      while (this._segments.size > this.maxSegments) {
        const oldest = this.list()[0];
        this._segments.delete(oldest.segment_id);
      }
    }

    /** Drop finalized segments older than the TTL; returns true if anything was removed. */
    prune() {
      const t = this.now();
      let removed = false;
      for (const [id, s] of this._segments) {
        if (s.msg.is_final && t - s.updatedAt > this.finalTtlMs) {
          this._segments.delete(id);
          removed = true;
        }
      }
      return removed;
    }

    list() {
      return Array.from(this._segments.values())
        .sort((a, b) => a.seq - b.seq)
        .map((s) => s.msg);
    }

    clear() {
      this._segments.clear();
    }
  }

  /** Decide which texts to show. prefs = { mode: 'both'|'translated'|'original', lang: 'es' }. */
  function pickText(msg, prefs) {
    const mode = MODES.indexOf(prefs.mode) >= 0 ? prefs.mode : 'both';
    const wantsTranslation = !!prefs.lang && prefs.lang !== msg.source_lang;
    const translated = wantsTranslation ? msg.translations[prefs.lang] || null : null;
    const pending = wantsTranslation && !translated; // translation requested but not here yet
    if (mode === 'original') return { original: msg.text, translated: null, pending: false };
    if (mode === 'translated') {
      return translated
        ? { original: null, translated: translated, pending: false }
        : { original: msg.text, translated: null, pending: pending }; // fall back so the user sees something
    }
    return { original: msg.text, translated: translated, pending: pending };
  }

  /** Renders store contents into a container, updating existing nodes in place. */
  class SubtitleRenderer {
    constructor(container, prefs) {
      this.container = container;
      this.prefs = Object.assign({ mode: 'both', lang: '' }, prefs || {});
      this.container.setAttribute('aria-live', 'polite');
      this.container.classList.add('subtitles');
    }

    setPrefs(prefs) {
      this.prefs = Object.assign({}, this.prefs, prefs);
    }

    render(segments) {
      const doc = this.container.ownerDocument;
      const keep = new Set();
      segments.forEach((msg, index) => {
        keep.add(msg.segment_id);
        let el = this.container.querySelector('[data-segment-id="' + cssEscape(msg.segment_id) + '"]');
        if (!el) {
          el = doc.createElement('div');
          el.className = 'subtitle';
          el.setAttribute('data-segment-id', msg.segment_id);
          for (const cls of ['speaker', 'original', 'translated']) {
            const span = doc.createElement('span');
            span.className = cls;
            el.appendChild(span);
          }
        }
        const shown = pickText(msg, this.prefs);
        el.setAttribute('data-final', msg.is_final ? 'true' : 'false');
        el.classList.toggle('partial', !msg.is_final);
        el.classList.toggle('pending', shown.pending);
        const speaker = el.querySelector('.speaker');
        const original = el.querySelector('.original');
        const translated = el.querySelector('.translated');
        speaker.textContent = msg.speaker;
        speaker.hidden = !msg.speaker;
        original.textContent = shown.original || '';
        original.hidden = !shown.original;
        original.setAttribute('lang', msg.source_lang);
        translated.textContent = shown.translated || '';
        translated.hidden = !shown.translated;
        if (this.prefs.lang) translated.setAttribute('lang', this.prefs.lang);
        // keep DOM order identical to store order
        if (this.container.children[index] !== el) this.container.insertBefore(el, this.container.children[index] || null);
      });
      Array.from(this.container.children).forEach((child) => {
        if (!keep.has(child.getAttribute('data-segment-id'))) child.remove();
      });
    }
  }

  function cssEscape(value) {
    return String(value).replace(/["\\]/g, '\\$&');
  }

  /** Subscribe to subtitle packets on a LiveKit room. Returns an unsubscribe function. */
  function attachDataChannel(room, RoomEvent, onSubtitle, onError) {
    const handler = function (payload, participant, kind, topic) {
      if (topic && topic !== SUBTITLE_TOPIC) return; // other data on the room
      try {
        onSubtitle(parseSubtitlePayload(payload), participant);
      } catch (err) {
        if (onError) onError(err);
      }
    };
    room.on(RoomEvent.DataReceived, handler);
    return function () {
      room.off(RoomEvent.DataReceived, handler);
    };
  }

  return {
    SUBTITLE_TOPIC,
    SCHEMA_VERSION,
    MODES,
    parseSubtitlePayload,
    SubtitleStore,
    SubtitleRenderer,
    pickText,
    attachDataChannel,
  };
});
