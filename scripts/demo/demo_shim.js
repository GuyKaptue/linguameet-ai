/* DEMO ONLY. Stand-in for the livekit-client UMD bundle: no network, no media. On connect() it replays the
 * subtitle packets recorded from the real pipeline (window.__SESSION__) through the same event names the
 * real client emits, so client/app.js and client/subtitles.js run unmodified. */
(function () {
  const speed = parseFloat(new URLSearchParams(location.search).get('speed') || '1');
  window.__attrs = [];
  class Room {
    constructor() { this.h = {}; this.canPlaybackAudio = true; this.localParticipant = { setAttributes: async (a) => { window.__attrs.push(a); } }; }
    on(e, f) { (this.h[e] = this.h[e] || []).push(f); return this; }
    off(e, f) { this.h[e] = (this.h[e] || []).filter((x) => x !== f); return this; }
    emit(e, ...a) { (this.h[e] || []).slice().forEach((f) => f(...a)); }
    async connect() {
      window.__replayStart = performance.now();
      const enc = new TextEncoder();
      window.__SESSION__.forEach(([t, json]) => setTimeout(() => this.emit('dataReceived', enc.encode(json), { identity: 'linguameet-gateway' }, 1, 'subtitles'), t / speed));
    }
    async disconnect() { this.emit('disconnected'); }
    async startAudio() {}
  }
  window.LivekitClient = {
    Room,
    RoomEvent: { DataReceived: 'dataReceived', TrackSubscribed: 'trackSubscribed', TrackUnsubscribed: 'trackUnsubscribed', Disconnected: 'disconnected', AudioPlaybackStatusChanged: 'audioPlaybackChanged' },
    Track: { Kind: { Audio: 'audio' } },
  };
})();
