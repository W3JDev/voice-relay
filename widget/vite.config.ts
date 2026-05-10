import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import dts from 'vite-plugin-dts';
import { resolve } from 'path';

export default defineConfig({
  plugins: [
    react(),
    dts({
      insertTypesEntry: true,
      include: ['src'],
      outDir: 'dist',
    }),
  ],
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'VoiceRelayWidget',
      formats: ['umd', 'es'],
      fileName: (format) => `voice-relay-widget.${format}.js`,
    },
    rollupOptions: {
      // React must be external so the host app's copy is used
      external: ['react', 'react-dom', 'react/jsx-runtime'],
      output: {
        // UMD globals – required when react/react-dom are external in UMD bundles
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM',
          'react/jsx-runtime': 'ReactJSXRuntime',
        },
        // Extract CSS into style.css
        assetFileNames: 'style.css',
      },
    },
    // Target modern browsers to keep the bundle lean
    target: 'es2020',
    // Keep readable class/function names for easier debugging
    minify: 'esbuild',
    sourcemap: true,
  },
});
