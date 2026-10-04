import { useEffect, useMemo, useRef, useState } from "react";
import { WS_URL } from "./config";

export type SimEvent = { type: string; t?: number; [k: string]: unknown };

export type KnowledgeCoverage = { filled: number; confirmed: number; total: number };

/** 3-second forecast from the model-predictive safety system (backend/sim/predictor.py). */
export type Prediction = {
  risk: "none" | "low" | "medium" | "high";
  conflict?: { hazard: string; in_s: number; dist: number; crash?: boolean };
  action?: string;
  action_label?: string;
  road_in_s?: number;
  road_alt?: number;
  stop_dist: number;
  cannot_stop: boolean;
  path: number[][];
  stop_point: number[] | null;
};

export type Guardian = { enabled: boolean; armed: boolean; engaged: boolean; action: string | null; interventions: number };

export type AIUsage = { calls: number; tokens: number; cached_tokens: number; cost_usd: number; by_purpose: Record<string, number> };

export type SimState = {
  t: number;
  pos: number[];
  vel: number[];
  acc: number[];
  speed: number;
  horizontal_speed?: number;
  vertical_speed?: number;
  altitude: number;
  yaw: number;
  heading_deg?: number;
  rpy: number[];
  quat: number[];
  cable_dist: number;
  cable_dz?: number;
  pylon_dist?: number;
  tree_dist?: number;
  road_dist?: number;
  insulator_dist?: number;
  nearest_insulator?: string;
  nearest_cable_point: number[];
  collided: boolean;
  collision_with?: string | null;
  wind_speed?: number;
  wind_from_deg?: number;
  compass_interference?: number;
  position_hold?: boolean;
  defects_spotted?: string[];
  mode: string;
  inspected_count: number;
  session_active?: boolean;
  task?: string | null; // competence-grid task the pilot is doing
  task_name?: string | null;
  knowledge?: KnowledgeCoverage;
  prediction?: Prediction | null;
  guardian?: Guardian;
  ai_usage?: AIUsage;
};

export type AIQuestion = {
  question: string;
  slot?: string | null; // competence-grid slot the question tries to fill
  slot_name?: string | null;
  kind?: "rule" | "hypothesis" | "deviation" | "follow_up";
  observation?: string;
  event: SimEvent;
  telemetry: Partial<SimState>;
  t: number;
};

export type AIObservation = { t: number; observation: string; asked: boolean };

/** What the apprentice's attention model is thinking (backend/llm/attention.py). */
export type Attention = {
  score: number;
  ready: boolean;
  reasons: string[];
  targets: Array<{ slot: string; name: string; kind: string }>;
  cooldown_s: number;
  thinking: boolean;
  waiting_answer: boolean;
  follow_up: boolean;
  pilot_talking?: boolean;
  off_record?: boolean;
};

/** The expert's rule for a situation, in their words, with the moment of their flight it was taught at. */
export type ExpertMoment = {
  slot: string;
  slot_name: string;
  rule: string;
  reason?: string;
  quote?: string | null;
  question?: string | null;
  phase?: string | null;
  session?: string | null;
  t?: number | null;
  frame: boolean;
};

export type AIAdvice = {
  speech: string;
  category: "safety_alert" | "technique_tip" | "qa_response";
  urgency: "low" | "medium" | "high";
  knowledge_reference: string;
  caught?: string; // the situation of a mistake the tutor caught before it happened
  replay?: ExpertMoment; // the expert's moment to show with it
};

/** Novice tutor: "what would the expert do here?" (predict) or "why would the expert not...?" (why). */
export type PredictQuestion = {
  id: string;
  topic: string;
  slot: string;
  slot_name: string;
  kind: "predict" | "why";
  question: string;
  t: number;
};

export function useSimSocket() {
  const [state, setState] = useState<SimState | null>(null);
  const [events, setEvents] = useState<SimEvent[]>([]);
  const [latestQuestion, setLatestQuestion] = useState<AIQuestion | null>(null);
  const [latestAdvice, setLatestAdvice] = useState<AIAdvice | null>(null);
  const [latestObservation, setLatestObservation] = useState<AIObservation | null>(null);
  const [attention, setAttention] = useState<Attention | null>(null);
  const [latestPrediction, setLatestPrediction] = useState<PredictQuestion | null>(null);
  const [offRecord, setOffRecord] = useState(false);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    let closed = false;

    function connect() {
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);

      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data);
          if (data.type === "state") {
            setState(data);
            if (!data.session_active) setOffRecord(false);
          }
          else if (data.type === "event") setEvents((prev) => [...prev.slice(-99), data.event]);
          else if (data.type === "question") setLatestQuestion(data);
          else if (data.type === "advice") setLatestAdvice(data);
          else if (data.type === "observation") setLatestObservation(data);
          else if (data.type === "attention") setAttention(data);
          else if (data.type === "predict") setLatestPrediction(data);
          else if (data.type === "off_record") setOffRecord(!!data.active);
        } catch (e) {
          console.error("WS parse error:", e);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (!closed) reconnectTimeout = setTimeout(connect, 2000);
      };
    }

    connect();

    return () => {
      closed = true;
      clearTimeout(reconnectTimeout);
      ws?.close();
    };
  }, []);

  const api = useMemo(
    () => ({
      state,
      events,
      latestQuestion,
      latestAdvice,
      latestObservation,
      attention,
      latestPrediction,
      offRecord,
      connected,
      /** Voice-activity detection: the pilot started or stopped talking (nothing else is sent). */
      sendVoiceActivity: (speaking: boolean) => {
        const ws = wsRef.current;
        if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "voice", speaking }));
      },
      sendKeys: (down: string[]) => {
        const ws = wsRef.current;
        if (ws?.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "keys", down }));
        }
      },
      sendFrame: (frame_b64: string, t: number) => {
        const ws = wsRef.current;
        if (ws?.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "frame", frame_b64, t }));
        }
      },
    }),
    [events, state, latestQuestion, latestAdvice, latestObservation, attention, latestPrediction, offRecord, connected]
  );

  return api;
}
