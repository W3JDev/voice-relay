/**
 * useAudioPlayback
 *
 * Queue-based audio playback with crossfade between chunks.
 * Handles Int16 PCM (base64-encoded) as sent by the Voice Relay server.
 */

import { useCallback, useRef, useState } from 'react';
import type { UseAudioPlaybackReturn } from '../types';

interface QueueEntry {
  data: string; // base64 Int16 PCM
  sampleRate: number;
}

export function useAudioPlayback(): UseAudioPlaybackReturn {
  const [isPlaying, setIsPlaying] = useState(false);

  // We keep all mutable playback state in refs so callbacks don't go stale
  const ctxRef = useRef<AudioContext | null>(null);
  const queueRef = useRef<QueueEntry[]>([]);
  const playingRef = useRef(false);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const currentGainRef = useRef<GainNode | null>(null);

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  function ensureContext(sampleRate: number): AudioContext {
    if (!ctxRef.current || ctxRef.current.state === 'closed') {
      ctxRef.current = new (
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      )({ sampleRate });
    }
    if (ctxRef.current.state === 'suspended') {
      ctxRef.current.resume();
    }
    return ctxRef.current;
  }

  function base64ToInt16(b64: string): Int16Array {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new Int16Array(bytes.buffer);
  }

  function int16ToFloat32(int16: Int16Array): Float32Array {
    const f32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      f32[i] = int16[i] / 32768.0;
    }
    return f32;
  }

  const playNext = useCallback(() => {
    if (queueRef.current.length === 0) {
      playingRef.current = false;
      currentSourceRef.current = null;
      currentGainRef.current = null;
      setIsPlaying(false);
      return;
    }

    playingRef.current = true;
    setIsPlaying(true);

    const chunk = queueRef.current.shift()!;

    try {
      const int16 = base64ToInt16(chunk.data);
      const float32 = int16ToFloat32(int16);

      const ctx = ensureContext(chunk.sampleRate);
      const buffer = ctx.createBuffer(1, float32.length, chunk.sampleRate);
      // TS 5.6+ made Float32Array generic; copyToChannel expects Float32Array<ArrayBuffer>
      buffer.copyToChannel(float32 as unknown as Float32Array<ArrayBuffer>, 0);

      // Gain node for crossfade / barge-in muting
      const gainNode = ctx.createGain();
      const now = ctx.currentTime;
      const duration = buffer.duration;
      const FADE = 0.015; // 15 ms crossfade

      gainNode.gain.setValueAtTime(0, now);
      gainNode.gain.linearRampToValueAtTime(1.0, now + FADE);
      if (duration > FADE * 2) {
        gainNode.gain.setValueAtTime(1.0, now + duration - FADE);
        gainNode.gain.linearRampToValueAtTime(0.0, now + duration);
      }
      gainNode.connect(ctx.destination);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(gainNode);

      currentSourceRef.current = source;
      currentGainRef.current = gainNode;

      source.onended = () => {
        if (currentSourceRef.current === source) {
          currentSourceRef.current = null;
          currentGainRef.current = null;
        }
        playNext();
      };

      source.start(0);
    } catch (err) {
      console.error('[VoiceRelay] Audio playback error:', err);
      // Skip this chunk and continue
      playNext();
    }
  }, []);

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /** Add a base64 Int16 PCM chunk to the playback queue */
  const enqueue = useCallback(
    (base64Data: string, sampleRate: number) => {
      // Pre-warm the AudioContext on first enqueue (may be called from a
      // user-gesture callback so we can unlock the context here if needed)
      ensureContext(sampleRate);

      queueRef.current.push({ data: base64Data, sampleRate });
      if (!playingRef.current) {
        playNext();
      }
    },
    [playNext],
  );

  /** Immediately silence all audio and clear the queue (barge-in) */
  const stopAll = useCallback(() => {
    queueRef.current = [];
    playingRef.current = false;

    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.stop();
      } catch (_) {
        // already stopped
      }
      currentSourceRef.current = null;
    }

    if (currentGainRef.current && ctxRef.current) {
      try {
        currentGainRef.current.gain.setValueAtTime(0, ctxRef.current.currentTime);
      } catch (_) {
        // context may be closing
      }
      currentGainRef.current = null;
    }

    setIsPlaying(false);
  }, []);

  return { isPlaying, enqueue, stopAll };
}
