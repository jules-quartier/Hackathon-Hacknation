// Voice-activity detection during a flight: is the pilot talking right now?
// The microphone stays open (echo cancellation on) but nothing is recorded or sent: only "talking"
// and "silent" transitions go to the backend, so the apprentice and the tutor never talk over the
// pilot. Suppressed while the AI speaks or an answer is being recorded (those are not "talking over").

const TALK_LEVEL = 0.025; // RMS above this counts as voice
const START_MS = 300; // this long above the level: the pilot started talking
const STOP_MS = 900; // this long below it: the pilot stopped

export type VoiceActivity = {
  stop: () => void;
  setSuppressed: (suppressed: boolean) => void;
  speaking: () => boolean;
};

export async function startVoiceActivity(onChange: (speaking: boolean) => void): Promise<VoiceActivity | null> {
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch {
    return null; // no microphone: the AI only waits for the stick inputs and its own timing
  }
  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  ctx.createMediaStreamSource(stream).connect(analyser);
  const samples = new Float32Array(analyser.fftSize);

  let speaking = false;
  let suppressed = false;
  let aboveSince: number | null = null;
  let belowSince: number | null = null;

  const set = (v: boolean) => {
    if (v === speaking) return;
    speaking = v;
    onChange(v);
  };

  const timer = setInterval(() => {
    if (suppressed) {
      aboveSince = belowSince = null;
      set(false);
      return;
    }
    analyser.getFloatTimeDomainData(samples);
    const rms = Math.sqrt(samples.reduce((s, v) => s + v * v, 0) / samples.length);
    const now = Date.now();
    if (rms > TALK_LEVEL) {
      belowSince = null;
      aboveSince = aboveSince ?? now;
      if (!speaking && now - aboveSince >= START_MS) set(true);
    } else {
      aboveSince = null;
      belowSince = belowSince ?? now;
      if (speaking && now - belowSince >= STOP_MS) set(false);
    }
  }, 100);

  return {
    stop: () => {
      clearInterval(timer);
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
      set(false);
    },
    setSuppressed: (v: boolean) => {
      suppressed = v;
    },
    speaking: () => speaking,
  };
}
