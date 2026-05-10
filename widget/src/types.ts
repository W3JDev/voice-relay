import type { CSSProperties } from 'react';

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

export type ConnectionState = 'connecting' | 'connected' | 'reconnecting' | 'disconnected';

export interface TranscriptEntry {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  partial: boolean;
  timestamp: number;
}

export interface VoiceRelayProps {
  /** WebSocket URL, e.g. wss://voice-relay.example.com/ws */
  url: string;
  /** Optional API key – appended as ?api_key=… query param */
  apiKey?: string;
  /** Color theme (default: 'dark') */
  theme?: 'dark' | 'light';
  /** Show only the floating mic button, no transcript panel */
  compact?: boolean;
  /** Widget position. 'inline' renders in document flow; the others are fixed overlays */
  position?: 'inline' | 'bottom-right' | 'bottom-left';
  /** Show conversation transcript panel (default: true) */
  showTranscript?: boolean;
  /** Enable continuous listening – automatically re-starts after each AI response */
  continuous?: boolean;

  // ---- Lifecycle callbacks ------------------------------------------------
  onSessionStart?: (sessionId: string) => void;
  /** Fired for every transcript event. `final` is true on the confirmed utterance. */
  onTranscript?: (text: string, final: boolean) => void;
  /** Fired when the AI response text is fully assembled */
  onResponse?: (text: string) => void;
  /** Fired on WebSocket or audio errors */
  onError?: (error: string) => void;

  // ---- Style overrides ----------------------------------------------------
  className?: string;
  style?: CSSProperties;
}

// ---------------------------------------------------------------------------
// Internal WebSocket message types
// ---------------------------------------------------------------------------

// Outbound (client → server)
export interface WsMsgAudio {
  type: 'audio';
  data: string; // base64 float32 PCM @ 16 kHz
}
export interface WsMsgPing {
  type: 'ping';
}
export interface WsMsgStartListening {
  type: 'start_listening';
}
export interface WsMsgStopListening {
  type: 'stop_listening';
}
export interface WsMsgReconnect {
  type: 'reconnect';
  session_id: string;
}
export interface WsMsgConfig {
  type: 'config';
  [key: string]: unknown;
}

export type WsOutbound =
  | WsMsgAudio
  | WsMsgPing
  | WsMsgStartListening
  | WsMsgStopListening
  | WsMsgReconnect
  | WsMsgConfig;

// Inbound (server → client)
export interface WsEvtSessionStart {
  type: 'session_start';
  session_id?: string;
}
export interface WsEvtListeningStarted {
  type: 'listening_started';
}
export interface WsEvtListeningStopped {
  type: 'listening_stopped';
}
export interface WsEvtVadStatus {
  type: 'vad_status';
  speech_detected: boolean;
}
export interface WsEvtTranscript {
  type: 'transcript';
  text: string;
  final: boolean;
}
export interface WsEvtResponseText {
  type: 'response_text';
  text: string;
}
export interface WsEvtResponseChunk {
  type: 'response_chunk';
  text: string;
}
export interface WsEvtAudioChunk {
  type: 'audio_chunk';
  data: string; // base64 Int16 PCM
  sample_rate?: number;
}
export interface WsEvtAudioResponse {
  type: 'audio_response';
  data: string;
  sample_rate?: number;
}
export interface WsEvtResponseComplete {
  type: 'response_complete';
  text?: string;
}
export interface WsEvtInterrupted {
  type: 'interrupted';
}
export interface WsEvtPong {
  type: 'pong';
}
export interface WsEvtError {
  type: 'error';
  message?: string;
}

export type WsInbound =
  | WsEvtSessionStart
  | WsEvtListeningStarted
  | WsEvtListeningStopped
  | WsEvtVadStatus
  | WsEvtTranscript
  | WsEvtResponseText
  | WsEvtResponseChunk
  | WsEvtAudioChunk
  | WsEvtAudioResponse
  | WsEvtResponseComplete
  | WsEvtInterrupted
  | WsEvtPong
  | WsEvtError;

// ---------------------------------------------------------------------------
// Hook return shapes
// ---------------------------------------------------------------------------

export interface UseVoiceRelayReturn {
  connectionState: ConnectionState;
  isRecording: boolean;
  isPlaying: boolean;
  vadActive: boolean;
  sessionId: string | null;
  transcript: TranscriptEntry[];
  status: string;
  error: string | null;
  bargeInVisible: boolean;
  /** Live PCM amplitude data for waveform rendering */
  waveformData: Float32Array;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  toggleRecording: () => Promise<void>;
  clearTranscript: () => void;
}

export interface UseAudioPlaybackReturn {
  isPlaying: boolean;
  enqueue: (base64Data: string, sampleRate: number) => void;
  stopAll: () => void;
}

export interface UseAudioRecorderReturn {
  isRecording: boolean;
  waveformData: Float32Array;
  start: () => Promise<void>;
  stop: () => void;
}
