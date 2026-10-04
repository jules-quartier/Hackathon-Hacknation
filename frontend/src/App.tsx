import { useEffect, useRef, useState } from "react";
import { ExpertMomentCard } from "./ExpertMoment";
import { FlightComparison } from "./FlightComparison";
import { Hud } from "./Hud";
import { KnowledgeViewer } from "./KnowledgeViewer";
import { MiniMap } from "./MiniMap";
import { Scene3D, SceneData } from "./Scene3D";
import { VoicePanel } from "./VoicePanel";
import { WorkMapView } from "./WorkMapView";
import { ExpertMoment, useSimSocket } from "./useSimSocket";
import { primeMicrophone } from "./recordAnswer";
import { API_URL } from "./config";

const FRAME_MAX_WIDTH = 768;
const TRAIL_SPACING_M = 0.5; // new path point once the drone has moved this far
const TRAIL_MAX_POINTS = 6000;
const FLIGHT_KEYS = ["arrowup", "arrowdown", "arrowleft", "arrowright", " ", "shift", "w", "s", "a", "d", "q", "e"];

type Tab = "ai" | "knowledge" | "debrief";

function Logo() {
  return (
    <div className="logo">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <path d="M13.5 2 5 13.5h6L9.5 22 19 9.5h-6.2L13.5 2z" fill="#3be8ff" />
      </svg>
    </div>
  );
}

function clock(t: number) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `T+${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function App() {
  const [mode, setMode] = useState<"expert" | "novice">("expert");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [scene, setScene] = useState<SceneData | null>(null);
  const [keysDown, setKeysDown] = useState<Set<string>>(new Set());
  const [inspected, setInspected] = useState<Set<string>>(new Set());
  const [tab, setTab] = useState<Tab>("ai");
  const [kbRefreshKey, setKbRefreshKey] = useState<number>(0);
  const [guardianOn, setGuardianOn] = useState(true);
  const [replay, setReplay] = useState<ExpertMoment | null>(null); // the expert's moment shown when the tutor steps in
  const [workMapKey, setWorkMapKey] = useState(0);

  const {
    state,
    events,
    latestQuestion,
    latestAdvice,
    latestObservation,
    attention,
    latestPrediction,
    offRecord,
    connected,
    sendKeys,
    sendFrame,
    sendVoiceActivity,
  } = useSimSocket();
  const lastFrameSendTime = useRef<number>(0);

  // Flight path since take-off: points spaced TRAIL_SPACING_M apart, cleared on Start or sim reset
  const [trail, setTrail] = useState<number[][]>([]);
  const lastTrailT = useRef<number>(0);
  useEffect(() => {
    if (!state?.pos) return;
    const t = state.t;
    const p = state.pos;
    const reset = t < lastTrailT.current;
    lastTrailT.current = t;
    setTrail((prev) => {
      const base = reset ? [] : prev;
      const last = base[base.length - 1];
      if (last && Math.hypot(p[0] - last[0], p[1] - last[1], p[2] - last[2]) < TRAIL_SPACING_M) return base;
      return [...base.slice(-(TRAIL_MAX_POINTS - 1)), [p[0], p[1], p[2]]];
    });
  }, [state]);

  // Load 3D scene data (defects are re-drawn on every flight, so it is reloaded on Start)
  const loadScene = () =>
    fetch(`${API_URL}/scene`)
      .then((r) => r.json())
      .then(setScene)
      .catch((error) => console.warn("Backend unavailable: scene could not be loaded.", error));
  useEffect(() => {
    loadScene();
  }, []);

  // Insulators turn green in 3D, on the radar and on the map as they are inspected
  useEffect(() => {
    const e = events[events.length - 1];
    if (e?.type === "insulator_inspected" && e.insulator_id) {
      setInspected((prev) => new Set([...prev, String(e.insulator_id)]));
    }
  }, [events]);

  // Periodic camera frame capture from the 3D canvas, for the LLM observer and tutor.
  // state/sendFrame change at ~30 Hz, so read them through refs: depending on them would reset the timer forever.
  const latestState = useRef(state);
  latestState.current = state;
  const sendFrameRef = useRef(sendFrame);
  sendFrameRef.current = sendFrame;
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      if (now - lastFrameSendTime.current < 2000) return;
      lastFrameSendTime.current = now;

      const canvas = document.querySelector<HTMLCanvasElement>("section.viewport canvas");
      const s = latestState.current;
      if (!canvas || !s || !canvas.width) return;
      try {
        // Downscale before sending: smaller payload and fewer image tokens for Claude
        const scale = Math.min(1, FRAME_MAX_WIDTH / canvas.width);
        const small = document.createElement("canvas");
        small.width = Math.round(canvas.width * scale);
        small.height = Math.round(canvas.height * scale);
        small.getContext("2d")?.drawImage(canvas, 0, 0, small.width, small.height);
        sendFrameRef.current(small.toDataURL("image/jpeg", 0.7), s.t);
      } catch {
        // Canvas may be tainted or rendering
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Refresh the knowledge tab when the apprentice learns or confirms a rule (confirmations happen without an answer)
  const kb = state?.knowledge;
  useEffect(() => {
    if (kb) setKbRefreshKey((k) => k + 1);
  }, [kb?.filled, kb?.confirmed]);

  const startSession = async () => {
    // Ask for the microphone once here (a click is required); it is only switched on after each question.
    if (!(await primeMicrophone())) {
      console.warn("Microphone blocked: answers can still be typed.");
    }
    try {
      await fetch(`${API_URL}/guardian`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: guardianOn }),
      }).catch(() => {});
      const res = await fetch(`${API_URL}/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      const body = await res.json();
      setSessionId(body.session_id);
      setInspected(new Set());
      setTrail([]);
      setReplay(null);
      setTab("ai");
      loadScene();
      (document.querySelector("section.viewport") as HTMLElement | null)?.focus();
    } catch (err) {
      console.error("Failed to start session:", err);
    }
  };

  const stopSession = async () => {
    try {
      const res = await fetch(`${API_URL}/session/stop`, { method: "POST" });
      if (!res.ok) return;
      const body = await res.json();
      setSessionId(body.session_id);
      setWorkMapKey((k) => k + 1);
      // expert: the spoken debrief runs in the Apprentice tab, then opens the Work Map; novice: straight to the debrief
      if (body.mode !== "expert") setTab("debrief");
    } catch (err) {
      console.error("Failed to stop session:", err);
    }
  };

  const toggleGuardian = async (enabled: boolean) => {
    setGuardianOn(enabled);
    await fetch(`${API_URL}/guardian`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    }).catch(() => {});
  };

  // Keyboard controls synchronization
  useEffect(() => {
    sendKeys(Array.from(keysDown));
  }, [keysDown, sendKeys]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "TEXTAREA" || (e.target as HTMLElement)?.tagName === "INPUT") return;
      const key = e.key.toLowerCase();
      if (FLIGHT_KEYS.includes(key)) e.preventDefault();
      setKeysDown((prev) => (prev.has(key) ? prev : new Set([...prev, key])));
    };
    const onKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      setKeysDown((prev) => {
        if (!prev.has(key)) return prev;
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    };
    const onBlur = () => setKeysDown(new Set());

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
    };
  }, []);

  // the backend is the source of truth: true from Start Flight until End Flight
  const sessionActive = !!state?.session_active;
  const pos = state?.pos ?? [-10, -10, 0];
  const yaw = state?.yaw ?? state?.rpy?.[2] ?? 0;
  const usage = state?.ai_usage;
  const defectsFound = (state?.defects_spotted ?? []).map((id) => {
    const d = scene?.defects?.find((x) => x.id === id);
    return d ? `${d.label} (${d.target})` : id;
  });

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <Logo />
          <div>
            <h1>AI APPRENTICE</h1>
          </div>
        </div>

        <div className="seg">
          <button className={mode === "expert" ? "active" : ""} onClick={() => setMode("expert")} disabled={sessionActive}>
            Expert
            <small>AI learns</small>
          </button>
          <button className={mode === "novice" ? "active" : ""} onClick={() => setMode("novice")} disabled={sessionActive}>
            Novice
            <small>AI coaches</small>
          </button>
        </div>

        {mode === "novice" && (
          <label className="switch" title="Predictive collision avoidance for novices">
            <input type="checkbox" checked={guardianOn} onChange={(e) => toggleGuardian(e.target.checked)} />
            <span className="track" />
            Guardian
          </label>
        )}

        <div className="topbar-spacer" />

        <div className="status-strip">
          <span className={`chip ${connected ? "ok live" : "bad"}`}>
            <span className="dot" />
            {connected ? "Link" : "Offline"}
          </span>
          <span className="chip ai" title={usage ? `${usage.tokens} tokens, ${usage.cached_tokens} from cache` : ""}>
            <span className="dot" />
            AI {usage?.calls ?? 0} · ${(usage?.cost_usd ?? 0).toFixed(3)}
          </span>
        </div>
        <div className="flight-clock">{sessionActive ? clock(state?.t ?? 0) : "T+--:--"}</div>
        <button className="btn btn-launch" onClick={startSession}>
          ▶ {sessionActive ? "Restart" : "Start flight"}
        </button>
        <button className="btn btn-abort" onClick={stopSession} disabled={!sessionActive}>
          ■ End
        </button>
      </header>

      <main className="workspace">
        <section className="viewport" tabIndex={0}>
          <Scene3D
            scene={scene}
            pos={pos}
            yaw={yaw}
            inspected={inspected}
            trail={trail}
            prediction={sessionActive ? state?.prediction : null}
            cablePoint={state?.nearest_cable_point}
            cableDist={state?.cable_dist}
          />
          <Hud state={state} scene={scene} mode={mode} inspected={inspected} defectsFound={defectsFound} sessionActive={sessionActive} />
          {offRecord && sessionActive && <div className="offrec-badge">● OFF THE RECORD</div>}
          {replay && <ExpertMomentCard moment={replay} onClose={() => setReplay(null)} />}
          <div className="hud-card minimap-card" style={{ pointerEvents: "none" }}>
            <div className="map-title">
              <span className="eyebrow">Tactical map</span>
              <span className="dim mono" style={{ fontSize: 10 }}>
                {pos[0].toFixed(0)}, {pos[1].toFixed(0)}
              </span>
            </div>
            <MiniMap scene={scene} pos={pos} yaw={yaw} inspected={inspected} trail={trail} prediction={sessionActive ? state?.prediction : null} />
          </div>
        </section>

        <aside className="sidebar">
          <div className="tabs">
            <button className={`tab ${tab === "ai" ? "active" : ""}`} onClick={() => setTab("ai")}>
              {mode === "expert" ? "Apprentice" : "Tutor"}
            </button>
            <button className={`tab ${tab === "knowledge" ? "active" : ""}`} onClick={() => setTab("knowledge")}>
              Knowledge
              {kb && <span className="tab-badge">{kb.filled}/{kb.total}</span>}
            </button>
            <button className={`tab ${tab === "debrief" ? "active" : ""}`} onClick={() => setTab("debrief")}>
              Debrief
            </button>
          </div>

          <div className="tab-body">
            {/* always mounted so End Flight can stop its audio/voice session even from another tab */}
            <VoicePanel
              mode={mode}
              sessionId={sessionId}
              simTime={state?.t ?? 0}
              latestQuestion={latestQuestion}
              latestAdvice={latestAdvice}
              latestObservation={latestObservation}
              attention={attention}
              prediction={state?.prediction}
              sessionActive={sessionActive}
              hidden={tab !== "ai"}
              latestPrediction={latestPrediction}
              offRecord={offRecord}
              onKnowledgeUpdated={() => setKbRefreshKey((k) => k + 1)}
              onReplay={setReplay}
              onDebriefDone={() => {
                setWorkMapKey((k) => k + 1);
                setTab("debrief");
              }}
              sendVoiceActivity={sendVoiceActivity}
            />
            {tab === "knowledge" && <KnowledgeViewer refreshKey={kbRefreshKey} currentTask={state?.task} />}
            {tab === "debrief" && (
              <div className="debrief">
                {mode === "expert" && <WorkMapView sessionId={sessionId} refreshKey={workMapKey} />}
                <FlightComparison currentSessionId={sessionId} />
                {mode === "novice" && <WorkMapView sessionId={null} refreshKey={workMapKey} />}
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}
