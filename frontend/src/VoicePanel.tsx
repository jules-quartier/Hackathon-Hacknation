import { useEffect, useRef, useState } from "react";
import { AIAdvice, AIObservation, AIQuestion, Attention, ExpertMoment, PredictQuestion, Prediction } from "./useSimSocket";
import { API_URL } from "./config";
import { AnswerRecording, NOTE_LIMITS, audioContext, beep, recordAnswer } from "./recordAnswer";
import { VoiceActivity, startVoiceActivity } from "./voiceActivity";
import { VoiceViz } from "./VoiceViz";

type DebriefItem = {
  id: number;
  slot: string;
  slot_name: string;
  kind: string;
  guardrail: boolean;
  question: string;
  why: string;
  t: number | null;
  status: "pending" | "answered" | "skipped";
  answer?: string | null;
  learned?: string | null;
};
type TeachBackState = {
  speech: string;
  slots: string[];
  round: number;
  confirmed: boolean;
  corrections?: Array<{ slot: string | null; slot_name?: string | null; said: string; learned?: string | null }>;
};
export type DebriefState = {
  session_id: string;
  phase: "questions" | "teachback" | "done";
  intro: string;
  done_when: string;
  items: DebriefItem[];
  teach_back: TeachBackState | null;
};
type Reply = { audio?: Blob; text?: string } | null;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const DEBRIEF_KIND: Record<string, string> = {
  deviation: "Rule check",
  unanswered: "Unanswered",
  hypothesis: "Habit check",
  rule: "Gap",
  unsure: "Exception",
  unseen: "Not seen",
  follow_up: "Follow-up",
};
const VERDICT_LABEL: Record<string, string> = { right: "Right", partly: "Almost", wrong: "Not quite" };

type Role = "ai" | "pilot" | "tutor" | "system" | "learned";
type TranscriptRow = { role: Role; text: string; t?: number; tag?: string; high?: boolean; ref?: string };
type QuestionMeta = { slotName?: string | null; kind?: AIQuestion["kind"] };
type AnswerResult = {
  insight?: string | null;
  rejected?: boolean;
  slot_name?: string | null;
  follow_up?: string;
  error?: string;
};

const KIND_LABEL: Record<string, string> = {
  rule: "New rule",
  hypothesis: "Habit check",
  deviation: "Rule check",
  follow_up: "Follow-up",
};
const ROLE_LABEL: Record<Role, string> = { ai: "Apprentice", pilot: "You", tutor: "Tutor", system: "System", learned: "✓ Rule learned" };

function useTypewriter(text: string, charsPerSecond = 60): string {
  const [n, setN] = useState(0);
  useEffect(() => {
    setN(0);
    if (!text) return;
    const id = setInterval(() => {
      setN((v) => {
        if (v >= text.length) {
          clearInterval(id);
          return v;
        }
        return v + 1;
      });
    }, 1000 / charsPerSecond);
    return () => clearInterval(id);
  }, [text]);
  return text.slice(0, n);
}

// Voice loop (expert mode): Claude's question is spoken with ElevenLabs text-to-speech, then the mic
// opens for one answer only, closes on silence, and the audio is transcribed by ElevenLabs on the backend.
// The pilot can also record a note on their own (button or R): the apprentice holds its questions meanwhile.
export function VoicePanel(props: {
  mode: "expert" | "novice";
  sessionId: string | null;
  simTime: number;
  latestQuestion: AIQuestion | null;
  latestAdvice: AIAdvice | null;
  latestObservation?: AIObservation | null;
  attention?: Attention | null;
  prediction?: Prediction | null;
  sessionActive: boolean;
  hidden?: boolean;
  latestPrediction?: PredictQuestion | null;
  offRecord?: boolean;
  onKnowledgeUpdated?: () => void;
  onReplay?: (m: ExpertMoment) => void;
  onDebriefDone?: (sessionId: string) => void;
  sendVoiceActivity?: (speaking: boolean) => void;
}) {
  const {
    mode,
    sessionId,
    simTime,
    latestQuestion,
    latestAdvice,
    latestObservation,
    attention,
    sessionActive,
    hidden,
    latestPrediction,
    offRecord = false,
    onKnowledgeUpdated,
    onReplay,
    onDebriefDone,
    sendVoiceActivity,
  } = props;

  const [transcript, setTranscript] = useState<TranscriptRow[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState<string>("");
  const [answerInput, setAnswerInput] = useState<string>("");
  const [noviceQueryInput, setNoviceQueryInput] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [showDetails, setShowDetails] = useState<boolean>(false);
  const [audioPlaying, setAudioPlaying] = useState<boolean>(false);
  // what the open mic records; "reply" = a debrief answer (longer pauses allowed)
  const [recording, setRecording] = useState<"answer" | "note" | "question" | "reply" | null>(null);
  const listening = recording !== null;
  const typed = useTypewriter(currentQuestion);

  const [pilotTalking, setPilotTalking] = useState(false);
  const [debrief, setDebriefState] = useState<DebriefState | null>(null);
  const [debriefRunning, setDebriefRunning] = useState(false);
  const [awaitingReply, setAwaitingReply] = useState<"item" | "teachback" | "predict" | null>(null);

  const lastQuestionRef = useRef<string>("");
  const lastPredictionRef = useRef<string>("");
  const pilotTalkingRef = useRef(false);
  const vadRef = useRef<VoiceActivity | null>(null);
  const debriefRef = useRef<DebriefState | null>(null);
  const debriefGenRef = useRef(0); // bumped to stop the debrief loop
  const typedReplyRef = useRef<((text: string) => void) | null>(null); // typed answer for the reply being awaited
  const micFailedRef = useRef(false);
  const typingInsteadRef = useRef(false); // "Type instead": the open mic is closed, the typed answer is awaited
  const listenAgainRef = useRef<(() => void) | null>(null);
  const awaitingRef = useRef<"item" | "teachback" | "predict" | null>(null);
  awaitingRef.current = awaitingReply;
  const sendVoiceRef = useRef(sendVoiceActivity);
  sendVoiceRef.current = sendVoiceActivity;
  const answerInputRef = useRef("");
  answerInputRef.current = answerInput;
  const setDebrief = (d: DebriefState | null) => {
    debriefRef.current = d;
    setDebriefState(d);
  };
  const lastAdviceRef = useRef<AIAdvice | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const speakAnalyser = useRef<AnalyserNode | null>(null);
  const speechGenRef = useRef(0); // bumped to cancel speech that is still being fetched
  const playbackEndRef = useRef<((finished: boolean) => void) | null>(null);
  const recordingRef = useRef<AnswerRecording | null>(null);
  const noteActiveRef = useRef(false); // from Record a note until the note is sent or discarded
  const wasActiveRef = useRef(sessionActive);
  const logRef = useRef<HTMLDivElement | null>(null);
  // read by the async voice loop, which outlives the render it started in
  const liveRef = useRef({ sessionActive, currentQuestion });
  liveRef.current = { sessionActive, currentQuestion };

  const addRow = (row: TranscriptRow) => setTranscript((prev) => [...prev.slice(-60), row]);
  useEffect(() => {
    const el = logRef.current; // scroll the log itself, not the whole side panel
    if (el) el.scrollTop = el.scrollHeight;
  }, [transcript.length]);

  // ---- speaking (ElevenLabs text-to-speech, browser voice as fallback) ----
  const stopSpeaking = () => {
    speechGenRef.current += 1;
    audioRef.current?.pause();
    audioRef.current = null;
    speakAnalyser.current = null;
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    playbackEndRef.current?.(false);
    playbackEndRef.current = null;
    setAudioPlaying(false);
  };

  /** Route the clip through an analyser for the visualiser (only when audio is unlocked, else play it plainly). */
  const analyse = (audio: HTMLAudioElement) => {
    const ctx = audioContext();
    if (!ctx || ctx.state !== "running") return;
    try {
      const source = ctx.createMediaElementSource(audio);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      analyser.connect(ctx.destination);
      speakAnalyser.current = analyser;
    } catch {
      speakAnalyser.current = null;
    }
  };

  /** Resolves true once the text has been fully spoken, false if stopped or replaced. */
  const speakText = async (text: string): Promise<boolean> => {
    if (!text) return false;
    stopSpeaking(); // never talk over the previous clip
    const gen = speechGenRef.current;
    const finished = new Promise<boolean>((resolve) => {
      playbackEndRef.current = resolve;
    });
    const end = (ok: boolean) => {
      if (gen !== speechGenRef.current) return;
      playbackEndRef.current?.(ok);
      playbackEndRef.current = null;
      speakAnalyser.current = null;
      setAudioPlaying(false);
    };
    setAudioPlaying(true);
    try {
      const res = await fetch(`${API_URL}/elevenlabs/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`TTS HTTP ${res.status}`);
      const blob = await res.blob();
      if (gen !== speechGenRef.current) return false; // stopped or replaced while fetching
      const audio = new Audio(URL.createObjectURL(blob));
      audioRef.current = audio;
      analyse(audio);
      audio.onended = () => end(true);
      audio.onerror = () => end(false);
      await audio.play();
    } catch {
      if (gen !== speechGenRef.current) return false;
      if ("speechSynthesis" in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onend = () => end(true);
        utterance.onerror = () => end(false);
        window.speechSynthesis.speak(utterance);
      } else {
        end(false);
      }
    }
    return finished;
  };

  const voiceLevel = () => {
    if (recordingRef.current) return recordingRef.current.level();
    const an = speakAnalyser.current;
    if (!an) return audioPlaying ? 0.25 + 0.25 * Math.random() : 0;
    const buf = new Float32Array(an.fftSize);
    an.getFloatTimeDomainData(buf);
    return Math.min(1, Math.sqrt(buf.reduce((s, v) => s + v * v, 0) / buf.length) * 6);
  };

  // ---- answers ----
  const showAnswerResult = (question: string, answer: string, data: AnswerResult) => {
    addRow({ role: "pilot", text: answer, t: simTime, tag: question ? "Answer" : "Note" });
    if (data.error) {
      addRow({ role: "system", text: `Not saved: the knowledge model failed (${data.error}).` });
    } else if (data.rejected) {
      addRow({
        role: "system",
        text: data.follow_up
          ? "Not saved yet: too vague. A follow-up question will come at a calm moment."
          : "Not saved: no usable know-how in that answer.",
      });
    } else if (data.insight) {
      addRow({ role: "learned", text: data.insight, tag: data.slot_name ?? undefined });
      if (data.follow_up) addRow({ role: "system", text: "Saved; a follow-up question will make it more precise." });
      onKnowledgeUpdated?.();
    }
    setCurrentQuestion((q) => (q === question ? "" : q)); // answered; wait for the next one
  };

  const skipQuestion = async (question: string, reason: string) => {
    await fetch(`${API_URL}/dialogue/skip`, { method: "POST" }).catch(() => {});
    setCurrentQuestion((q) => (q === question ? "" : q));
    addRow({ role: "system", text: reason });
  };

  /** Beep, open the mic and wait until silence, Stop (Enter) or Discard (Esc). Null = nothing kept. */
  const record = async (kind: "answer" | "note" | "question" | "reply"): Promise<Blob | null> => {
    let rec: AnswerRecording;
    try {
      beep();
      await new Promise((r) => setTimeout(r, 250)); // don't record the beep
      rec = await recordAnswer(kind === "note" || kind === "reply" ? NOTE_LIMITS : undefined);
    } catch (err) {
      micFailedRef.current = true;
      addRow({ role: "system", text: `Microphone unavailable (${(err as Error)?.message ?? err}). Type it instead.` });
      return null;
    }
    recordingRef.current = rec;
    setRecording(kind);
    const audio = await rec.done;
    setRecording(null);
    if (recordingRef.current === rec) recordingRef.current = null;
    return audio;
  };

  /** The pilot is talking: wait until they stop before the AI says anything (at most maxMs). */
  const waitForSilence = async (maxMs = 15000) => {
    const start = Date.now();
    while (pilotTalkingRef.current && Date.now() - start < maxMs) await sleep(200);
  };

  /** One reply to what was just said: spoken (mic until silence) or typed in the input bar, whichever comes first. */
  const getReply = async (kind: "item" | "teachback" | "predict"): Promise<Reply> => {
    setAwaitingReply(kind);
    const typed = new Promise<Reply>((resolve) => {
      typedReplyRef.current = (text: string) => resolve({ text });
    });
    try {
      micFailedRef.current = false;
      typingInsteadRef.current = false;
      const spoken = record(kind === "predict" ? "answer" : "reply").then((audio): Reply => (audio ? { audio } : null));
      let reply = await Promise.race([typed, spoken]);
      if (reply?.text !== undefined) {
        recordingRef.current?.cancel();
      } else if (!reply && (micFailedRef.current || typingInsteadRef.current || answerInputRef.current.trim())) {
        reply = await typed; // no microphone, or the pilot chose to type: wait for the typed answer
      }
      return reply;
    } finally {
      typedReplyRef.current = null;
      typingInsteadRef.current = false;
      setAwaitingReply(null);
    }
  };

  /** Nothing heard: wait for a typed answer, a button, or 🎙 to listen again. */
  const waitForInput = (kind: "item" | "teachback" | "predict") =>
    new Promise<Reply | "listen">((resolve) => {
      setAwaitingReply(kind);
      typedReplyRef.current = (text: string) => resolve({ text });
      listenAgainRef.current = () => resolve("listen");
    }).finally(() => {
      typedReplyRef.current = null;
      listenAgainRef.current = null;
      setAwaitingReply(null);
    });

  /** Close the open mic and answer by typing instead. */
  const typeInstead = () => {
    typingInsteadRef.current = true;
    recordingRef.current?.cancel();
    setTimeout(() => document.querySelector<HTMLInputElement>(".input-bar .field")?.focus(), 50);
  };

  const post = async (path: string, body?: unknown, audio?: Blob) => {
    const res = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": audio ? audio.type || "audio/webm" : "application/json" },
      body: audio ?? (body === undefined ? undefined : JSON.stringify(body)),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
    return data;
  };

  /** Transcribe and learn from a recording; an empty question means the pilot's own note. */
  const sendVoice = async (audio: Blob, question: string) => {
    setIsSubmitting(true);
    try {
      const res = await fetch(`${API_URL}/dialogue/voice-answer?question=${encodeURIComponent(question)}`, {
        method: "POST",
        headers: { "Content-Type": audio.type || "audio/webm" },
        body: audio,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      if (!data.transcript) {
        addRow({ role: "system", text: question ? "Nothing understood in the recording: question skipped." : "Nothing understood in the note." });
        if (question) setCurrentQuestion((q) => (q === question ? "" : q));
      } else {
        showAnswerResult(question, data.transcript, data);
      }
    } catch (err) {
      addRow({ role: "system", text: `Could not process the recording: ${(err as Error)?.message ?? err}` });
    } finally {
      setIsSubmitting(false);
    }
  };

  /** Mic on for one answer, then off. Enter / Stop = done, Esc / Discard = skip, silence = done. */
  const listenForAnswer = async (question: string) => {
    const audio = await record("answer");
    // flight ended, a new question arrived, a note was started, or the answer was typed meanwhile
    if (!liveRef.current.sessionActive || liveRef.current.currentQuestion !== question) return;
    if (!audio) {
      await skipQuestion(question, "No answer recorded: question skipped.");
      return;
    }
    await sendVoice(audio, question);
  };

  /** The pilot's own note, without a question. The apprentice holds its questions until it is sent. */
  const recordNote = async () => {
    if (!liveRef.current.sessionActive || recordingRef.current || noteActiveRef.current) return;
    noteActiveRef.current = true;
    try {
      await recordAndSendNote();
    } finally {
      noteActiveRef.current = false;
    }
  };
  const recordAndSendNote = async () => {
    stopSpeaking();
    const dropped = liveRef.current.currentQuestion;
    liveRef.current.currentQuestion = "";
    setCurrentQuestion("");
    if (dropped) addRow({ role: "system", text: "Question dropped: recording your note instead." });
    await fetch(`${API_URL}/dialogue/pilot-note?active=true`, { method: "POST" }).catch(() => {});

    const audio = await record("note");
    if (!audio || !liveRef.current.sessionActive) {
      await fetch(`${API_URL}/dialogue/pilot-note?active=false`, { method: "POST" }).catch(() => {});
      if (liveRef.current.sessionActive) addRow({ role: "system", text: "Note discarded." });
      return;
    }
    await sendVoice(audio, "");
  };
  const recordNoteRef = useRef(recordNote);
  recordNoteRef.current = recordNote;

  /** Novice: show the pilot's question and speak the tutor's answer. */
  const showTutorAnswer = (query: string, advice: Partial<AIAdvice>) => {
    addRow({ role: "pilot", text: query, t: simTime, tag: "Question" });
    if (advice.speech) {
      addRow({ role: "tutor", text: advice.speech, t: simTime, tag: "Answer", ref: advice.knowledge_reference ?? undefined });
      speakText(advice.speech);
    }
  };

  /** Novice: ask the tutor out loud. Mic on until silence (Enter = send, Esc = cancel), then a spoken answer. */
  const askTutorByVoice = async () => {
    if (recordingRef.current || noteActiveRef.current) return;
    stopSpeaking(); // don't record the tutor's own voice
    const audio = await record("question");
    if (!audio) {
      addRow({ role: "system", text: "No question recorded." });
      return;
    }
    setIsSubmitting(true);
    try {
      const res = await fetch(`${API_URL}/dialogue/voice-advise`, {
        method: "POST",
        headers: { "Content-Type": audio.type || "audio/webm" },
        body: audio,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
      if (!data.transcript) {
        addRow({ role: "system", text: "Nothing understood: ask again or type your question." });
      } else {
        showTutorAnswer(data.transcript, data);
      }
    } catch (err) {
      addRow({ role: "system", text: `Could not ask the tutor: ${(err as Error)?.message ?? err}` });
    } finally {
      setIsSubmitting(false);
    }
  };
  const askTutorByVoiceRef = useRef(askTutorByVoice);
  askTutorByVoiceRef.current = askTutorByVoice;

  /** New question: show it, speak it, then listen for the answer. */
  const askAndListen = async (question: string, t?: number, meta: QuestionMeta = {}) => {
    recordingRef.current?.cancel();
    lastQuestionRef.current = question;
    liveRef.current.currentQuestion = question;
    setCurrentQuestion(question);
    addRow({ role: "ai", text: question, t, tag: KIND_LABEL[meta.kind ?? "rule"] ?? "Question" });
    await waitForSilence(); // never talk over the pilot
    if (liveRef.current.currentQuestion !== question) return;
    const spoken = await speakText(question);
    if (spoken && mode === "expert" && liveRef.current.sessionActive && liveRef.current.currentQuestion === question) {
      await listenForAnswer(question);
    }
  };

  // R (during a flight, not typing): record a note in expert mode, ask the tutor by voice in novice mode
  useEffect(() => {
    if (!sessionActive) return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || e.repeat) return;
      if (e.key.toLowerCase() === "r") (mode === "expert" ? recordNoteRef : askTutorByVoiceRef).current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, sessionActive]);

  // Enter / Esc while the mic is open
  useEffect(() => {
    if (!listening) return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === "Enter") recordingRef.current?.finish();
      if (e.key === "Escape") recordingRef.current?.cancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [listening]);

  // Question pushed by the observer
  useEffect(() => {
    if (latestQuestion?.question && latestQuestion.question !== lastQuestionRef.current) {
      lastQuestionRef.current = latestQuestion.question;
      // a question sent just before Record a note: the backend already dropped it
      if (sessionActive && !noteActiveRef.current) {
        askAndListen(latestQuestion.question, latestQuestion.t, {
          slotName: latestQuestion.slot_name,
          kind: latestQuestion.kind,
        });
      }
    }
  }, [latestQuestion]);

  // Tutor advice (novice mode): spoken only. Compared by message, not text: a repeated alarm has the same words.
  useEffect(() => {
    if (latestAdvice?.speech && latestAdvice !== lastAdviceRef.current) {
      lastAdviceRef.current = latestAdvice;
      if (!sessionActive) return;
      addRow({
        role: "tutor",
        text: latestAdvice.speech,
        t: simTime,
        tag: latestAdvice.category.replace("_", " "),
        high: latestAdvice.urgency === "high",
        ref: latestAdvice.knowledge_reference ?? undefined,
      });
      // the tutor caught a mistake: show the expert's rule, words and screen moment with it
      if (latestAdvice.replay) onReplay?.(latestAdvice.replay);
      if (awaitingRef.current === "predict") recordingRef.current?.cancel(); // the backend dropped that question
      if (recordingRef.current) {
        if (latestAdvice.urgency !== "high") return; // shown, not spoken into the open mic
        recordingRef.current.cancel(); // safety first: drop the question being recorded
      }
      const advice = latestAdvice;
      void (async () => {
        if (advice.urgency === "low") await waitForSilence(8000); // a tip waits for the pilot to finish talking
        if (lastAdviceRef.current === advice && liveRef.current.sessionActive) speakText(advice.speech);
      })();
    }
  }, [latestAdvice]);

  // ---- voice-activity detection: the AI never talks over the pilot ----
  useEffect(() => {
    if (!sessionActive) return;
    let cancelled = false;
    startVoiceActivity((speaking) => {
      pilotTalkingRef.current = speaking;
      setPilotTalking(speaking);
      sendVoiceRef.current?.(speaking);
    }).then((vad) => {
      if (cancelled) vad?.stop();
      else vadRef.current = vad;
    });
    return () => {
      cancelled = true;
      vadRef.current?.stop();
      vadRef.current = null;
      pilotTalkingRef.current = false;
      setPilotTalking(false);
    };
  }, [sessionActive]);
  // the AI's own voice and the answers being recorded are not "the pilot talking over"
  useEffect(() => {
    vadRef.current?.setSuppressed(audioPlaying || recording !== null);
  }, [audioPlaying, recording]);

  // ---- novice: "what would the expert do here?" ----
  const askPrediction = async (q: PredictQuestion) => {
    if (recordingRef.current) return; // busy asking the tutor: the backend times the question out
    addRow({ role: "tutor", text: q.question, t: q.t, tag: q.kind === "why" ? "Why?" : "Predict", ref: q.slot_name });
    await waitForSilence(8000);
    if (!liveRef.current.sessionActive) return;
    const spoken = await speakText(q.question);
    let reply: Reply = null;
    if (spoken && liveRef.current.sessionActive) reply = await getReply("predict");
    if (!liveRef.current.sessionActive) return;
    if (!reply) {
      await post("/teach/predict-skip", { id: q.id }).catch(() => {});
      addRow({ role: "system", text: "No answer: the tutor carries on." });
      return;
    }
    setIsSubmitting(true);
    try {
      const data = reply.audio
        ? await post(`/teach/voice-predict-answer?id=${q.id}`, undefined, reply.audio)
        : await post("/teach/predict-answer", { id: q.id, answer: reply.text });
      if (!data.speech) {
        addRow({ role: "system", text: "Nothing understood: the tutor carries on." });
        return;
      }
      addRow({ role: "pilot", text: data.transcript ?? reply.text ?? "", tag: q.kind === "why" ? "Your reason" : "Your prediction" });
      addRow({ role: "tutor", text: data.speech, tag: VERDICT_LABEL[data.verdict] ?? "Feedback", high: data.verdict === "wrong", ref: data.slot_name });
      if (data.replay) onReplay?.(data.replay);
      speakText(data.speech);
    } catch (err) {
      addRow({ role: "system", text: `The tutor could not judge the answer: ${(err as Error)?.message ?? err}` });
    } finally {
      setIsSubmitting(false);
    }
  };
  useEffect(() => {
    const q = latestPrediction;
    if (!q || q.id === lastPredictionRef.current) return;
    lastPredictionRef.current = q.id;
    if (sessionActive && mode === "novice") void askPrediction(q);
  }, [latestPrediction]);

  // ---- expert: off the record ----
  const toggleOffRecord = async () => {
    const active = !offRecord;
    if (active) {
      stopSpeaking();
      recordingRef.current?.cancel();
      liveRef.current.currentQuestion = "";
      setCurrentQuestion("");
    }
    try {
      await post(`/dialogue/off-record?active=${active}`);
      addRow({
        role: "system",
        text: active
          ? "Off the record: no questions, no camera frames, nothing learned until you switch back."
          : "Back on the record.",
        t: simTime,
      });
    } catch (err) {
      addRow({ role: "system", text: `Could not switch: ${(err as Error)?.message ?? err}` });
    }
  };

  // ---- expert: spoken debrief after the flight (gap questions, then the teach-back) ----
  const stopDebrief = () => {
    debriefGenRef.current += 1;
    typedReplyRef.current = null;
    listenAgainRef.current = null;
    recordingRef.current?.cancel();
    stopSpeaking();
    setDebriefRunning(false);
    setCurrentQuestion("");
  };

  const askDebriefItem = async (sid: string, item: DebriefItem, alive: () => boolean) => {
    const items = debriefRef.current?.items ?? [];
    const pos = items.findIndex((i) => i.id === item.id) + 1;
    addRow({
      role: "ai",
      text: item.question,
      tag: `Debrief ${pos}/${items.length} · ${DEBRIEF_KIND[item.kind] ?? "Gap"}${item.guardrail ? " · guardrail" : ""}`,
      ref: item.slot_name,
    });
    setCurrentQuestion(item.question);
    await waitForSilence();
    await speakText(item.question);
    if (!alive()) return;
    const reply = await getReply("item");
    if (!alive()) return;
    setCurrentQuestion("");
    let state: DebriefState;
    if (!reply) {
      state = await post(`/debrief/${sid}/skip`, { item_id: item.id });
      addRow({ role: "system", text: "No answer: question skipped." });
    } else {
      setIsSubmitting(true);
      try {
        const data = reply.audio
          ? await post(`/debrief/${sid}/voice-answer?item_id=${item.id}`, undefined, reply.audio)
          : await post(`/debrief/${sid}/answer`, { item_id: item.id, answer: reply.text });
        if (!data.state) {
          state = await post(`/debrief/${sid}/skip`, { item_id: item.id });
          addRow({ role: "system", text: "Nothing understood: question skipped." });
        } else {
          state = data.state;
          const r: AnswerResult = data.result ?? {};
          addRow({ role: "pilot", text: data.transcript ?? reply.text ?? "", tag: "Answer" });
          if (r.insight) {
            addRow({ role: "learned", text: r.insight, tag: r.slot_name ?? undefined });
            onKnowledgeUpdated?.();
          } else {
            addRow({
              role: "system",
              text: r.error ? `Not saved: the knowledge model failed (${r.error}).` : "Not saved: no usable know-how in that answer.",
            });
          }
          if (r.follow_up) addRow({ role: "system", text: "One follow-up question to make it precise." });
        }
      } finally {
        setIsSubmitting(false);
      }
    }
    if (alive()) setDebrief(state);
  };

  const runDebrief = async (sid: string) => {
    const gen = ++debriefGenRef.current;
    const alive = () => gen === debriefGenRef.current;
    setDebriefRunning(true);
    try {
      let state: DebriefState = await post(`/debrief/${sid}/start`);
      if (!alive()) return;
      setDebrief(state);
      if (state.phase === "done") return;
      const resumed = state.items.some((i) => i.status !== "pending") || !!state.teach_back;
      const intro = resumed ? "Let's finish the debrief." : state.intro;
      addRow({ role: "ai", text: intro, tag: "Debrief" });
      await speakText(intro);
      let spokenRound = 0;
      while (alive()) {
        state = debriefRef.current!;
        if (state.phase === "questions") {
          const item = state.items.find((i) => i.status === "pending");
          if (!item) break;
          await askDebriefItem(sid, item, alive);
          continue;
        }
        if (state.phase !== "teachback") break;
        if (!state.teach_back) {
          addRow({ role: "system", text: "Gaps closed. The apprentice is preparing its teach-back…" });
          setIsSubmitting(true);
          try {
            state = await post(`/debrief/${sid}/teachback`);
          } finally {
            setIsSubmitting(false);
          }
          if (!alive()) return;
          setDebrief(state);
        }
        const tb = state.teach_back!;
        if (tb.round !== spokenRound) {
          spokenRound = tb.round;
          addRow({ role: "ai", text: tb.speech, tag: tb.round === 1 ? "Teach-back" : `Teach-back · round ${tb.round}` });
          await waitForSilence();
          await speakText(tb.speech);
          if (!alive()) return;
        }
        let reply = await getReply("teachback");
        while (alive() && !reply) {
          addRow({ role: "system", text: "Say “yes” or what to correct: type it, use the buttons, or press 🎙 to answer by voice." });
          const r = await waitForInput("teachback");
          reply = r === "listen" ? await getReply("teachback") : r;
        }
        if (!alive() || !reply) return;
        setIsSubmitting(true);
        let data;
        try {
          data = reply.audio
            ? await post(`/debrief/${sid}/teachback/voice-reply`, undefined, reply.audio)
            : await post(`/debrief/${sid}/teachback/reply`, { answer: reply.text });
        } finally {
          setIsSubmitting(false);
        }
        if (!alive()) return;
        if (!data.state) {
          addRow({ role: "system", text: "Nothing understood in the reply." });
          continue;
        }
        addRow({ role: "pilot", text: data.transcript ?? reply.text ?? "", tag: "Teach-back reply" });
        for (const c of data.corrected ?? []) {
          if (c.learned) addRow({ role: "learned", text: c.learned, tag: c.slot_name ?? undefined });
        }
        if (data.corrected?.length) onKnowledgeUpdated?.();
        setDebrief(data.state);
        if (data.state.phase === "done") {
          const msg = data.state.teach_back?.confirmed
            ? "Thank you, that is confirmed. The Work Map is ready for the next pilot."
            : "Thank you, your corrections are saved. The Work Map is ready.";
          addRow({ role: "ai", text: msg, tag: "Debrief complete" });
          onKnowledgeUpdated?.();
          await speakText(msg);
          if (alive()) onDebriefDone?.(sid);
          break;
        }
      }
    } catch (err) {
      if (alive()) addRow({ role: "system", text: `Debrief paused: ${(err as Error)?.message ?? err}` });
    } finally {
      if (gen === debriefGenRef.current) setDebriefRunning(false);
    }
  };

  // End Flight stops everything (speech, microphone, pending question), then the expert's debrief starts.
  // Start Flight stops a debrief still running.
  useEffect(() => {
    if (wasActiveRef.current && !sessionActive) {
      stopSpeaking();
      recordingRef.current?.cancel();
      setCurrentQuestion("");
      if (mode === "expert" && sessionId) {
        addRow({ role: "system", text: "Flight ended. The apprentice stops asking during flight and starts the debrief.", t: simTime });
        const sid = sessionId;
        setTimeout(() => void runDebrief(sid), 600);
      } else {
        addRow({ role: "system", text: "Flight ended. The tutor has stopped coaching.", t: simTime });
      }
    }
    if (!wasActiveRef.current && sessionActive) {
      stopDebrief();
      setDebrief(null);
    }
    wasActiveRef.current = sessionActive;
  }, [sessionActive]);

  // Expert Mode: Trigger manual question from AI
  const triggerQuestion = async () => {
    if (!sessionActive) return;
    try {
      const res = await fetch(`${API_URL}/dialogue/trigger-question`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "manual_inquiry", t: simTime }),
      });
      const data = await res.json();
      if (!res.ok) {
        addRow({ role: "system", text: data.detail ?? "Could not get a question from the apprentice." });
        return;
      }
      if (data.question && liveRef.current.sessionActive) {
        askAndListen(data.question, data.t, { slotName: data.slot_name, kind: data.kind });
      }
    } catch (err) {
      console.error("Failed to trigger question:", err);
    }
  };

  // Expert Mode: typed answer (also cancels the mic if it is open)
  const submitAnswer = async () => {
    const ans = answerInput.trim();
    if (!ans) return;
    if (typedReplyRef.current) {
      typedReplyRef.current(ans); // the debrief (or a prediction) is waiting for this answer
      setAnswerInput("");
      return;
    }
    if (!sessionActive) return;
    const question = currentQuestion;
    liveRef.current.currentQuestion = ""; // the voice loop must not also submit/skip this question
    recordingRef.current?.cancel();

    setIsSubmitting(true);
    try {
      const res = await fetch(`${API_URL}/dialogue/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question, answer: ans }), // no question: a note
      });
      const data = await res.json();
      if (data.ok) {
        setAnswerInput("");
        showAnswerResult(question, ans, data);
      }
    } catch (err) {
      console.error("Failed to submit answer:", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Novice Mode: Ask question to AI Tutor
  const askTutor = async () => {
    const query = noviceQueryInput.trim();
    if (!query) return;
    setNoviceQueryInput("");
    if (typedReplyRef.current) {
      typedReplyRef.current(query); // answering the tutor's "what would the expert do?"
      return;
    }

    try {
      const res = await fetch(`${API_URL}/dialogue/advise`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, query }),
      });
      showTutorAnswer(query, await res.json());
    } catch (err) {
      console.error("Failed to ask tutor:", err);
    }
  };

  // ---- view ----
  // one status line: the open mic and speech first, then what the apprentice is doing
  let status = {
    label: "Standby",
    orb: "idle",
    sub: mode === "expert" ? "Start a flight: the apprentice watches and learns." : "Start a flight: the tutor coaches you with the expert's rules.",
  };
  const deb = debrief;
  const tb = deb?.teach_back ?? null;
  const showDebrief = mode === "expert" && !sessionActive && !!deb;
  if (!sessionActive && deb && mode === "expert") {
    const answered = deb.items.filter((i) => i.status !== "pending").length;
    if (recording) status = { label: "Listening to you", orb: "", sub: "Enter = done · Esc = skip · or type it" };
    else if (isSubmitting) status = { label: "Learning…", orb: "thinking", sub: "Turning your answer into a rule." };
    else if (audioPlaying) status = { label: "Debrief", orb: "thinking", sub: deb.phase === "teachback" ? "Explaining it back to you." : "Asking." };
    else if (deb.phase === "done") status = { label: "Debrief complete", orb: "", sub: tb?.confirmed ? "Teach-back confirmed. The Work Map is ready." : "Corrections saved. The Work Map is ready." };
    else if (awaitingReply === "teachback") status = { label: "Is that right?", orb: "", sub: "Confirm, or say what to correct." };
    else if (deb.phase === "teachback") status = { label: "Teach-back", orb: "thinking", sub: "The apprentice explains the process back." };
    else if (debriefRunning) status = { label: "Debrief", orb: "", sub: `Gap ${Math.min(answered + 1, deb.items.length)} of ${deb.items.length}.` };
    else status = { label: "Debrief paused", orb: "idle", sub: `${answered} of ${deb.items.length} gaps closed.` };
  }
  if (sessionActive) {
    const a = attention;
    if (recording === "note") status = { label: "Recording your note", orb: "", sub: "Enter = save · Esc = discard" };
    else if (recording === "question") status = { label: "Listening to your question", orb: "", sub: "Enter = ask · Esc = cancel" };
    else if (recording === "answer" || recording === "reply") status = { label: "Listening to you", orb: "", sub: "Enter = done · Esc = skip" };
    else if (isSubmitting) status = { label: "Processing…", orb: "thinking", sub: "Transcribing and learning." };
    else if (audioPlaying) status = { label: "Speaking", orb: "thinking", sub: mode === "expert" ? "Question on air." : "Tutor on air." };
    else if (offRecord && mode === "expert") status = { label: "Off the record", orb: "idle", sub: "Nothing is asked, seen or learned until you switch back." };
    else if (pilotTalking) status = { label: "Listening quietly", orb: "idle", sub: mode === "expert" ? "You are talking: the apprentice waits." : "You are talking: the tutor waits." };
    else if (mode === "novice") status = { label: "Coaching", orb: "", sub: "Watching your flight with the expert's rules." };
    else if (a?.thinking) status = { label: "Thinking…", orb: "thinking", sub: "Composing a question for this moment." };
    else if (a?.waiting_answer) status = { label: "Waiting for your answer", orb: "", sub: "Speak or type it below." };
    else if (a?.follow_up) status = { label: "Follow-up queued", orb: "", sub: "Asked at the next calm moment." };
    else if (a?.cooldown_s) status = { label: "Observing", orb: "", sub: `Next question in ${a.cooldown_s}s at the earliest.` };
    else if (a?.ready) status = { label: "Ready to ask", orb: "thinking", sub: "Something worth learning just happened." };
    else status = { label: "Observing", orb: "", sub: "Watching your flight." };
  }
  const score = attention?.score ?? 0;
  const detailsAvailable = mode === "expert" && sessionActive;

  // the question being asked is the newest apprentice message: highlighted, typed out, replayable
  let currentRow = -1;
  let lastTutorRow = -1;
  transcript.forEach((r, i) => {
    if (r.role === "ai" && currentQuestion && r.text === currentQuestion) currentRow = i;
    if (r.role === "tutor") lastTutorRow = i;
  });

  const rowClass = (r: TranscriptRow, i: number) => `msg ${r.role}${r.high ? " high" : ""}${i === currentRow ? " current" : ""}`;

  return (
    <div className="ai-panel" style={hidden ? { display: "none" } : undefined}>
      <div className="panel status-line">
        <div className="status-row">
          <div className={`brain-orb ${status.orb}`} />
          <div className="status-text">
            <strong>{status.label}</strong>
            <span>{status.sub}</span>
          </div>
          {detailsAvailable && (
            <button className={`btn btn-sm btn-ghost ${showDetails ? "active" : ""}`} onClick={() => setShowDetails((v) => !v)}>
              Details
            </button>
          )}
          {mode === "expert" && sessionActive && (
            <button
              className={`btn btn-sm btn-ghost offrec ${offRecord ? "active" : ""}`}
              onClick={toggleOffRecord}
              title="Take the next part off the record: no questions, no frames, nothing learned"
            >
              {offRecord ? "● Off record" : "Off record"}
            </button>
          )}
          {mode === "expert" && sessionActive && (
            <button className="btn btn-sm" onClick={triggerQuestion} disabled={offRecord}>
              Ask now
            </button>
          )}
        </div>
        {detailsAvailable && showDetails && (
          <div className="details">
            <div className="attn-meter">
              <div className={`attn-fill ${attention?.ready ? "ready" : ""}`} style={{ width: `${Math.min(100, (score / 6) * 100)}%` }} />
              <div className="attn-threshold" style={{ left: "50%" }} />
            </div>
            <div className="attn-meta">
              <span>ATTENTION {score.toFixed(1)}</span>
              <span>ASK ≥ 3.0</span>
            </div>
            {attention?.reasons?.length ? (
              <ul className="why">
                {attention.reasons.slice(0, 3).map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            ) : null}
            {attention?.targets?.length ? (
              <div className="target-chips">
                {attention.targets.map((t) => (
                  <span key={t.slot} className={`tchip kind-${t.kind}`}>
                    <b>{t.kind === "hypothesis" ? "HABIT" : t.kind === "deviation" ? "CHECK" : "LEARN"}</b>
                    {t.name}
                  </span>
                ))}
              </div>
            ) : null}
            {latestObservation && <div className="observation">👁 {latestObservation.observation}</div>}
          </div>
        )}
      </div>

      {showDebrief && deb && (
        <div className="panel debrief-card">
          <div className="panel-head">
            <span className="eyebrow">Spoken debrief</span>
            <span className={`chip ${deb.phase === "done" ? "ok" : "ai"}`}>
              {deb.phase === "questions" ? "Closing gaps" : deb.phase === "teachback" ? "Teach-back" : "Done"}
            </span>
          </div>
          <div className="gap-track">
            {deb.items.map((it) => (
              <span
                key={it.id}
                className={`gap-dot ${it.status} ${it.guardrail ? "guard" : ""}`}
                title={`${it.slot_name}${it.guardrail ? " (guardrail)" : ""}: ${it.question}`}
              />
            ))}
            <span className="gap-count">
              {deb.items.filter((i) => i.status === "answered").length}/{deb.items.length} gaps closed
              {deb.items.some((i) => i.guardrail) ? " · ◆ = guardrail" : ""}
            </span>
          </div>
          {tb && (
            <div className={`teachback ${tb.confirmed ? "ok" : ""}`}>
              <div className="tb-head">
                Teach-back{tb.round > 1 ? ` · round ${tb.round}` : ""}
                {tb.confirmed && <b> ✓ confirmed by the expert</b>}
              </div>
              <p>{tb.speech}</p>
            </div>
          )}
          <div className="done-when">Done when {deb.done_when}.</div>
          <div className="row-actions">
            {awaitingReply === "teachback" && (
              <>
                <button className="btn btn-sm btn-primary" onClick={() => typedReplyRef.current?.("Yes, that is how I do it.")}>
                  ✓ Confirm
                </button>
                <button className="btn btn-sm" onClick={typeInstead}>
                  ✎ Correct
                </button>
              </>
            )}
            {debriefRunning ? (
              <button className="btn btn-sm btn-ghost" onClick={stopDebrief}>
                Pause debrief
              </button>
            ) : deb.phase !== "done" ? (
              <button className="btn btn-sm" onClick={() => void runDebrief(deb.session_id)}>
                ▶ Resume debrief
              </button>
            ) : null}
            {deb.phase === "done" && (
              <button className="btn btn-sm btn-primary" onClick={() => onDebriefDone?.(deb.session_id)}>
                Open the Work Map →
              </button>
            )}
          </div>
        </div>
      )}

      <div className="panel feed">
        <div className="panel-head">
          <span className="eyebrow">Conversation</span>
        </div>
        <div className="transcript" ref={logRef}>
          {transcript.length === 0 ? (
            <div className="empty-note">
              {mode === "expert"
                ? "The apprentice asks when something worth learning happens: a road crossing, a defect, a near miss, a habit it noticed."
                : "The tutor coaches you with the rules the expert taught, and warns you before a danger. Ask it anything below."}
            </div>
          ) : (
            transcript.map((row, i) => (
              <div key={i} className={rowClass(row, i)}>
                <div className="msg-meta">
                  <span>{ROLE_LABEL[row.role]}</span>
                  {row.tag && <span>· {row.tag}</span>}
                  {row.t !== undefined && <span className="t">{row.t.toFixed(1)}s</span>}
                </div>
                <div>
                  {i === currentRow ? typed : row.text}
                  {i === currentRow && typed.length < row.text.length && <span className="caret" />}
                </div>
                {row.ref && <div className="msg-ref">◆ {row.ref}</div>}
                {(i === currentRow || i === lastTutorRow) && (
                  <button className="msg-replay" onClick={() => speakText(row.text)} disabled={listening}>
                    ▶ Replay
                  </button>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      <div className={`panel input-bar ${listening ? "recording" : ""}`}>
        {listening ? (
          <>
            <span className="rec-dot" />
            <VoiceViz mode="listening" getLevel={voiceLevel} />
            <button className="btn btn-primary btn-sm" onClick={() => recordingRef.current?.finish()}>
              ⏹ {recording === "question" ? "Ask" : "Save"}
            </button>
            {awaitingReply && (
              <button className="btn btn-sm" onClick={typeInstead} title="Close the mic and type the answer">
                ⌨
              </button>
            )}
            <button className="btn btn-sm" onClick={() => recordingRef.current?.cancel()} title="Discard">
              ✕
            </button>
          </>
        ) : (
          <>
            <button
              className="btn btn-mic"
              onClick={() => {
                if (listenAgainRef.current) listenAgainRef.current(); // debrief: answer by voice after all
                else if (mode === "expert") void recordNote();
                else void askTutorByVoice();
              }}
              disabled={listenAgainRef.current ? false : !sessionActive || isSubmitting || (mode === "expert" && offRecord)}
              title={awaitingReply ? "Answer by voice" : mode === "expert" ? "Record a note (R)" : "Ask the tutor by voice (R)"}
            >
              🎙
            </button>
            {mode === "expert" ? (
              <>
                <input
                  className="field"
                  type="text"
                  placeholder={
                    awaitingReply === "teachback"
                      ? "Confirm, or type what to correct…"
                      : awaitingReply || currentQuestion
                        ? "Answer by voice, or type it here…"
                        : sessionActive
                          ? offRecord
                            ? "Off the record: nothing is saved"
                            : "Type a note for the apprentice…"
                          : "The debrief starts when a flight ends"
                  }
                  value={answerInput}
                  disabled={!sessionActive && !awaitingReply}
                  onChange={(e) => setAnswerInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && submitAnswer()}
                />
                <button
                  className="btn btn-primary"
                  onClick={() => submitAnswer()}
                  disabled={isSubmitting || !answerInput.trim() || (!sessionActive && !awaitingReply)}
                >
                  {isSubmitting ? "…" : "Send"}
                </button>
              </>
            ) : (
              <>
                <input
                  className="field"
                  type="text"
                  placeholder={awaitingReply === "predict" ? "What would the expert do? Say it or type it…" : "Ask the tutor (e.g. how high to cross the road?)"}
                  value={noviceQueryInput}
                  onChange={(e) => setNoviceQueryInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && askTutor()}
                />
                <button className="btn btn-primary" onClick={() => askTutor()} disabled={!noviceQueryInput.trim()}>
                  Send
                </button>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
