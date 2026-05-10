/**
 * @openclaw/voice-relay-widget
 *
 * Embeddable React voice widget for the Voice Relay server.
 *
 * Usage (npm):
 *   import { VoiceRelay } from '@openclaw/voice-relay-widget';
 *   import '@openclaw/voice-relay-widget/style.css';
 *   <VoiceRelay url="wss://..." apiKey="vr_live_..." />
 *
 * Usage (UMD / vanilla):
 *   VoiceRelayWidget.mount('#voice-widget', { url: 'wss://...', apiKey: 'vr_live_...' });
 */

export { VoiceRelay } from './VoiceRelay';
export { mount, unmount } from './mount';
export type { VoiceRelayProps, TranscriptEntry, ConnectionState } from './types';
