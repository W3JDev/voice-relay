/**
 * useVoiceRelay
 *
 * Central hook that wires together WebSocket communication, audio recording,
 * audio playback, transcript management, and reconnection logic.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  ConnectionState,
  TranscriptEntry,
  UseVoiceRelayReturn,
  VoiceRelayProps,
  WsInbound,
  WsOutbound,
} from '../types';
import { useAudioPlayback } from './useAudioPlayback';
import { useAudioRecorder } from './useAudioRecorder';

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function makeId(): string {
  return Math.random().toString(36).slice(2, 10);
}

function float32ToBase64(f32: Float32Array): string {
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function buildWsUrl(url: string, apiKey?: string): string {
  try {
    const u = new URL(url);
    if (apiKey) u.searchParams.set('api_key', apiKey);
    return u.toString();
  } catch {
    // If url is already well-formed with params just append
    if (apiKey) {
      const sep = url.includes('?') ? '&' : '?';
      return `${url}${sep}api_key=${encodeURIComponent(apiKey)}`;
    }
    return url;
  }
}

const MAX_RECONNECT_DELAY = 16_000;

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export interface UseVoiceRelayOptions
  extends Pick<
    VoiceRelayProps,
    | 'url'
    | 'apiKey'
    | 'continuous'
    | 'onSessionStart'
    | 'onTranscript'
    | 'onResponse'
    | 'onError'
  > {}

export function useVoiceRelay(opts: UseVoiceRelayOptions): UseVoiceRelayReturn {
  const {
    url,
    apiKey,
    continuous = false,
    onSessionStart,
    onTranscript,
    onResponse,
    onError,
  } = opts;

  // ---- State ---------------------------------------------------------------
  const [connectionState, setConnectionState] =
    useState<ConnectionState>('connecting');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [status, setStatus] = useState('Connecting…');
  const [error, setError] = useState<string | null>(null);
  const [bargeInVisible, setBargeInVisible] = useState(false);
  const [vadActive, setVadActive] = useState(false);

  // ---- Refs (mutable, not triggering re-render) ----------------------------
  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelayRef = useRef(1000);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const continuousRef = useRef(continuous);
  const streamingTextRef = useRef('');
  const bargeInTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep continuousRef in sync
  useEffect(() => {
    continuousRef.current = continuous;
  }, [continuous]);

  // ---- Sub-hooks -----------------------------------------------------------
  const { isPlaying, enqueue, stopAll } = useAudioPlayback();

  // PCM frame → WebSocket
  const handleFrame = useCallback(
    (pcmFloat32: Float32Array) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const base64 = float32ToBase64(pcmFloat32);
      wsSend({ type: 'audio', data: base64 });
    },
    [],
  );

  const { isRecording, waveformData, start: recStart, stop: recStop } =
    useAudioRecorder({ onFrame: handleFrame });

  // ---- WebSocket helpers ---------------------------------------------------
  function wsSend(msg: WsOutbound) {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }

  const showError = useCallback(
    (msg: string) => {
      setError(msg);
      onError?.(msg);
    },
    [onError],
  );

  // ---- Message handler -----------------------------------------------------
  const handleMessage = useCallback(
    (msg: WsInbound) => {
      switch (msg.type) {
        case 'session_start': {
          const sid = msg.session_id ?? null;
          sessionIdRef.current = sid;
          setSessionId(sid);
          if (sid) onSessionStart?.(sid);
          break;
        }

        case 'listening_started':
          setStatus(
            continuousRef.current
              ? 'Listening (continuous)…'
              : 'Listening…',
          );
          break;

        case 'listening_stopped':
          setStatus('Processing…');
          break;

        case 'vad_status':
          setVadActive(msg.speech_detected);
          break;

        case 'transcript':
          if (!msg.final) {
            // Partial – update last partial entry or create one
            setTranscript((prev) => {
              const last = prev[prev.length - 1];
              if (last?.partial && last.role === 'user') {
                return [
                  ...prev.slice(0, -1),
                  { ...last, text: msg.text },
                ];
              }
              return [
                ...prev,
                {
                  id: makeId(),
                  role: 'user',
                  text: msg.text,
                  partial: true,
                  timestamp: Date.now(),
                },
              ];
            });
            onTranscript?.(msg.text, false);
          } else {
            // Final – finalise partial entry
            setTranscript((prev) => {
              const last = prev[prev.length - 1];
              if (last?.partial && last.role === 'user') {
                return [
                  ...prev.slice(0, -1),
                  { ...last, text: msg.text, partial: false },
                ];
              }
              return [
                ...prev,
                {
                  id: makeId(),
                  role: 'user',
                  text: msg.text,
                  partial: false,
                  timestamp: Date.now(),
                },
              ];
            });
            // Reset streaming response state
            streamingTextRef.current = '';
            setStatus('Getting response…');
            onTranscript?.(msg.text, true);
          }
          break;

        case 'response_text':
          // Legacy non-streaming response
          setTranscript((prev) => [
            ...prev,
            {
              id: makeId(),
              role: 'assistant',
              text: msg.text,
              partial: false,
              timestamp: Date.now(),
            },
          ]);
          onResponse?.(msg.text);
          break;

        case 'response_chunk':
          streamingTextRef.current += msg.text;
          setTranscript((prev) => {
            const last = prev[prev.length - 1];
            if (last?.role === 'assistant' && last.partial) {
              return [
                ...prev.slice(0, -1),
                { ...last, text: streamingTextRef.current },
              ];
            }
            return [
              ...prev,
              {
                id: makeId(),
                role: 'assistant',
                text: streamingTextRef.current,
                partial: true,
                timestamp: Date.now(),
              },
            ];
          });
          setStatus('Speaking…');
          break;

        case 'audio_chunk':
          enqueue(msg.data, msg.sample_rate ?? 24000);
          break;

        case 'audio_response':
          enqueue(msg.data, msg.sample_rate ?? 24000);
          setStatus('Speaking…');
          break;

        case 'response_complete': {
          const finalText = msg.text;
          setTranscript((prev) => {
            const last = prev[prev.length - 1];
            if (last?.role === 'assistant') {
              return [
                ...prev.slice(0, -1),
                {
                  ...last,
                  text: finalText ?? last.text,
                  partial: false,
                },
              ];
            }
            return prev;
          });
          if (finalText) onResponse?.(finalText);

          // In continuous mode, restart listening after playback completes.
          // We rely on the isPlaying → false transition in an effect below.
          if (!continuousRef.current) setStatus('Ready');
          break;
        }

        case 'interrupted':
          stopAll();
          // Flash barge-in indicator
          setBargeInVisible(true);
          if (bargeInTimerRef.current) clearTimeout(bargeInTimerRef.current);
          bargeInTimerRef.current = setTimeout(
            () => setBargeInVisible(false),
            1600,
          );
          setStatus(
            continuousRef.current ? 'Listening (continuous)…' : 'Listening…',
          );
          break;

        case 'error':
          showError(msg.message ?? 'Server error');
          break;

        case 'pong':
          break;

        default:
          break;
      }
    },
    [enqueue, stopAll, showError, onSessionStart, onTranscript, onResponse],
  );

  // ---- WebSocket connect / reconnect --------------------------------------
  const connect = useCallback(() => {
    // Clean up any previous socket
    if (wsRef.current) {
      try { wsRef.current.close(1000); } catch (_) { /* ignore */ }
      wsRef.current = null;
    }

    setConnectionState('reconnecting');
    setError(null);

    const fullUrl = buildWsUrl(url, apiKey);
    let ws: WebSocket;
    try {
      ws = new WebSocket(fullUrl);
    } catch (e) {
      showError(`Invalid WebSocket URL: ${(e as Error).message}`);
      setConnectionState('disconnected');
      return;
    }

    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionState('connected');
      setError(null);
      setStatus('Connected');
      reconnectDelayRef.current = 1000;

      // Session reconnect
      if (sessionIdRef.current) {
        wsSend({ type: 'reconnect', session_id: sessionIdRef.current });
      }

      // Keep-alive ping
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = setInterval(() => {
        wsSend({ type: 'ping' });
      }, 30_000);
    };

    ws.onclose = (evt) => {
      setConnectionState('disconnected');
      recStop();
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }

      // Map well-known close codes to user-friendly messages
      if (evt.code === 4001) {
        showError('This server requires an API key.');
        setStatus('API key required');
        return;
      }
      if (evt.code === 4002) {
        showError('Your API key is invalid or expired.');
        setStatus('Invalid API key');
        return;
      }
      if (evt.code === 4003) {
        showError('Too many requests – please wait and try again.');
        setStatus('Rate limited');
        return;
      }

      // Exponential backoff reconnect for everything else
      const delay = reconnectDelayRef.current;
      setStatus(
        `Disconnected – reconnecting in ${Math.round(delay / 1000)}s…`,
      );
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(() => {
        connect();
      }, delay);
      reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_DELAY);
    };

    ws.onerror = () => {
      showError('WebSocket error – is the server running?');
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data as string) as WsInbound;
        handleMessage(msg);
      } catch (_) {
        /* non-JSON frame – ignore */
      }
    };
  }, [url, apiKey, handleMessage, recStop, showError]);

  // ---- Continuous mode: auto-restart after playback finishes --------------
  // We watch `isPlaying` via an effect. When it transitions false and we are
  // in continuous mode, we restart recording (with a small grace period).
  const wasPlayingRef = useRef(false);
  useEffect(() => {
    if (wasPlayingRef.current && !isPlaying && continuousRef.current) {
      setStatus('Ready to listen…');
      const t = setTimeout(() => {
        if (continuousRef.current && !isRecording) {
          startRecording();
        }
      }, 300);
      return () => clearTimeout(t);
    }
    wasPlayingRef.current = isPlaying;
    return undefined;
  }, [isPlaying]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- Mount / unmount ----------------------------------------------------
  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      if (bargeInTimerRef.current) clearTimeout(bargeInTimerRef.current);
      try { wsRef.current?.close(1000); } catch (_) { /* ignore */ }
      recStop();
    };
  }, [connect]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- Recording control --------------------------------------------------
  const startRecording = useCallback(async () => {
    if (isRecording) return;
    try {
      await recStart();
      wsSend({ type: 'start_listening' });
    } catch (err) {
      showError(`Microphone error: ${(err as Error).message}`);
    }
  }, [isRecording, recStart, showError]);

  const stopRecording = useCallback(() => {
    if (!isRecording) return;
    recStop();
    wsSend({ type: 'stop_listening' });
  }, [isRecording, recStop]);

  const toggleRecording = useCallback(async () => {
    if (isRecording) {
      stopRecording();
    } else {
      await startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  const clearTranscript = useCallback(() => {
    setTranscript([]);
    streamingTextRef.current = '';
  }, []);

  return {
    connectionState,
    isRecording,
    isPlaying,
    vadActive,
    sessionId,
    transcript,
    status,
    error,
    bargeInVisible,
    waveformData,
    startRecording,
    stopRecording,
    toggleRecording,
    clearTranscript,
  };
}
