# MCP EMS Ambient Pipeline

This package is a **stdio-based MCP server** that exposes tools for an EMS/dispatch audio pipeline:

**audio → transcript → entities → claim JSON (→ eval)**

The MCP server is written in TypeScript/Node and **shells out to a local Python CLI** (your `ems_pipeline` command). It does **not** host HTTP.

---

## How it works

An MCP client (Codex, Claude Desktop, etc.) launches this server as a subprocess and communicates with it over **stdin/stdout** using MCP’s tool protocol.

Flow:

MCP Client (agent runtime)
→ stdio
→ Node MCP server (this repo)
→ Python CLI (`ems_pipeline`)
→ outputs JSON files and returns structured results to the agent

---

## Tools exposed

This server registers tools via `server.tool(...)` (SDK auto-implements `tools/list` and `tools/call`):

- `ems.preprocess_audio`
- `ems.transcribe_audio`
- `ems.extract_entities`
- `ems.build_claim`
- `ems.run_pipeline` (end-to-end)
- `ems.eval`

Each tool returns a JSON payload as MCP `content: [{ type: "text", text: ... }]`.

---

## Prerequisites

### Node
- Node 18+ recommended (Node 20+ preferred)

### Python CLI contract

Your environment must provide a CLI named `ems_pipeline` **or** a Python module runnable via `python -m ems_pipeline`.

Expected commands:

- `ems_pipeline preprocess <audio_path> --out <json> [--bandpass] [--denoise] --chunk-seconds N --overlap-seconds N`
- `ems_pipeline transcribe <audio_path> --out <json> [--bandpass] [--denoise]`
- `ems_pipeline extract <transcript_json> --out <json>`
- `ems_pipeline build-claim <entities_json> --out <json>`
- `ems_pipeline eval --gold <gold.json> --pred <pred.json> --out <metrics.json>`

If you don’t implement `preprocess` as a separate command, you can either:
- add it (recommended), or
- adjust the MCP tool to call `transcribe` with preprocessing flags only.

---

## Install

```bash
npm install