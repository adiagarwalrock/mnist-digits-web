# MNIST ONNX Draw Visualizer

Pure frontend app for MNIST digit drawing + live ONNX inference.

Created by **Aditya Agarwal**
Website: `https://www.adityaagarwal.me`
GitHub: `https://github.com/adiagarwalrock`
Repo: `https://github.com/adiagarwalrock/mnist-digits-web`

## Features

- Draw a digit on a square canvas.
- Real-time prediction from `web/model/model.onnx`.
- Output bars for `0-9` probabilities.
- Network visualization with distinct layer sizes:
  - Input: derived from input feature size.
  - Hidden: computed as a different count (not equal to input/output).
  - Output: 10 nodes.

## Local Run (No Python)

Use any static server. Example:

```bash
npm install
npm run preview
```

Then open the printed local URL.

## Deploy to Cloudflare Workers

Files already prepared:

- `wrangler.toml`
- `worker.js`

Commands:

```bash
npm install
npm run deploy:cf
```

For local worker preview:

```bash
npm run dev:cf
```

## Deploy to Vercel

Files already prepared:

- `vercel.json`

Commands:

```bash
npm install
npm run deploy:vercel
```

For local Vercel preview:

```bash
npm run dev:vercel
```

## Model Path

- Default model URL: `./model/model.onnx`
- File location in repo: `web/model/model.onnx`
