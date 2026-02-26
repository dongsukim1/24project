import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

import {
  preprocessAudio,
  transcribeAudio,
  extractEntities,
  buildClaim,
  runPipeline,
  evalPred
} from "./tools.js";

// ---- Schemas (define as z.object so we can use z.infer for typing) ----
const CommonEnvSchema = z.object({
  pythonBin: z.string().default("python"),
  cli: z.string().default("ems_pipeline"),
  cliMode: z.enum(["bin", "module"]).default("bin"),
  workdir: z.string().optional()
});

const PreprocessSchema = z.object({
  audio_path: z.string(),
  out_path: z.string(),
  bandpass: z.boolean().default(false),
  denoise: z.boolean().default(false),
  chunk_seconds: z.number().default(30),
  overlap_seconds: z.number().default(2),
  env: CommonEnvSchema.default({ pythonBin: "python", cli: "ems_pipeline", cliMode: "bin" })
});

const TranscribeSchema = z.object({
  audio_path: z.string(),
  out_path: z.string(),
  bandpass: z.boolean().default(false),
  denoise: z.boolean().default(false),
  model: z.string().optional(),
  env: CommonEnvSchema.default({ pythonBin: "python", cli: "ems_pipeline", cliMode: "bin" })
});

const ExtractSchema = z.object({
  transcript_json: z.string(),
  out_path: z.string(),
  env: CommonEnvSchema.default({ pythonBin: "python", cli: "ems_pipeline", cliMode: "bin" })
});

const BuildClaimSchema = z.object({
  entities_json: z.string(),
  out_path: z.string(),
  env: CommonEnvSchema.default({ pythonBin: "python", cli: "ems_pipeline", cliMode: "bin" })
});

const RunPipelineSchema = z.object({
  audio_path: z.string(),
  out_dir: z.string(),
  bandpass: z.boolean().default(false),
  denoise: z.boolean().default(false),
  model: z.string().optional(),
  env: CommonEnvSchema.default({ pythonBin: "python", cli: "ems_pipeline", cliMode: "bin" })
});

const EvalSchema = z.object({
  gold_path: z.string(),
  pred_path: z.string(),
  out_path: z.string(),
  env: CommonEnvSchema.default({ pythonBin: "python", cli: "ems_pipeline", cliMode: "bin" })
});

// ---- Server ----
const server = new McpServer({ name: "ems-ambient-pipeline", version: "0.1.0" });

function ok(result: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
}
function err(e: unknown) {
  const msg = e instanceof Error ? (e.stack ?? e.message) : String(e);
  return { isError: true, content: [{ type: "text" as const, text: msg }] };
}

// IMPORTANT: server.tool expects a Zod RAW SHAPE, so pass `.shape`
server.tool(
  "ems.preprocess_audio",
  PreprocessSchema.shape,
  async (args: z.infer<typeof PreprocessSchema>) => {
    try {
      return ok(await preprocessAudio(args as any));
    } catch (e) {
      return err(e);
    }
  }
);

server.tool(
  "ems.transcribe_audio",
  TranscribeSchema.shape,
  async (args: z.infer<typeof TranscribeSchema>) => {
    try {
      return ok(await transcribeAudio(args as any));
    } catch (e) {
      return err(e);
    }
  }
);

server.tool(
  "ems.extract_entities",
  ExtractSchema.shape,
  async (args: z.infer<typeof ExtractSchema>) => {
    try {
      return ok(await extractEntities(args as any));
    } catch (e) {
      return err(e);
    }
  }
);

server.tool(
  "ems.build_claim",
  BuildClaimSchema.shape,
  async (args: z.infer<typeof BuildClaimSchema>) => {
    try {
      return ok(await buildClaim(args as any));
    } catch (e) {
      return err(e);
    }
  }
);

server.tool(
  "ems.run_pipeline",
  RunPipelineSchema.shape,
  async (args: z.infer<typeof RunPipelineSchema>) => {
    try {
      return ok(await runPipeline(args as any));
    } catch (e) {
      return err(e);
    }
  }
);

server.tool(
  "ems.eval",
  EvalSchema.shape,
  async (args: z.infer<typeof EvalSchema>) => {
    try {
      return ok(await evalPred(args as any));
    } catch (e) {
      return err(e);
    }
  }
);

await server.connect(new StdioServerTransport());