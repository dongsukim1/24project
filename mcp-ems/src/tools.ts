import { z } from "zod";
import { execCmd, assertOk } from "./util.js";
import fs from "node:fs/promises";

const CommonEnv = z
  .object({
    pythonBin: z.string().default("python"),
    // If your CLI is installed as a module entrypoint, set to "ems_pipeline".
    // If you want to run "python -m ems_pipeline", set cliMode to "module".
    cli: z.string().default("ems_pipeline"),
    cliMode: z.enum(["bin", "module"]).default("bin"),
    workdir: z.string().optional()
  })
  .default({ pythonBin: "python", cli: "ems_pipeline", cliMode: "bin" });

function cliCommand(env: z.infer<typeof CommonEnv>): { cmd: string; baseArgs: string[] } {
  if (env.cliMode === "module") return { cmd: env.pythonBin, baseArgs: ["-m", env.cli] };
  return { cmd: env.cli, baseArgs: [] };
}

async function readJson(path: string) {
  const txt = await fs.readFile(path, "utf8");
  return JSON.parse(txt);
}

export const ToolSchemas = {
  preprocess_audio: z.object({
    audio_path: z.string(),
    out_path: z.string(),
    bandpass: z.boolean().default(false),
    denoise: z.boolean().default(false),
    chunk_seconds: z.number().default(30),
    overlap_seconds: z.number().default(2),
    env: CommonEnv
  }),
  transcribe_audio: z.object({
    audio_path: z.string(),
    out_path: z.string(),
    bandpass: z.boolean().default(false),
    denoise: z.boolean().default(false),
    model: z.string().optional(), // e.g. "base", "small"
    env: CommonEnv
  }),
  extract_entities: z.object({
    transcript_json: z.string(),
    out_path: z.string(),
    env: CommonEnv
  }),
  build_claim: z.object({
    entities_json: z.string(),
    out_path: z.string(),
    env: CommonEnv
  }),
  run_pipeline: z.object({
    audio_path: z.string(),
    out_dir: z.string(),
    bandpass: z.boolean().default(false),
    denoise: z.boolean().default(false),
    model: z.string().optional(),
    env: CommonEnv
  }),
  eval: z.object({
    gold_path: z.string(),
    pred_path: z.string(),
    out_path: z.string(),
    env: CommonEnv
  })
};

export async function preprocessAudio(input: z.infer<typeof ToolSchemas.preprocess_audio>) {
  const env = input.env;
  const { cmd, baseArgs } = cliCommand(env);

  // You can implement this as "ems_pipeline preprocess" or fold into "transcribe" flags.
  // If you don't have a preprocess command yet, this tool can be a no-op wrapper or you can add it.
  const args = [
    ...baseArgs,
    "preprocess",
    input.audio_path,
    "--out",
    input.out_path,
    "--chunk-seconds",
    String(input.chunk_seconds),
    "--overlap-seconds",
    String(input.overlap_seconds)
  ];
  if (input.bandpass) args.push("--bandpass");
  if (input.denoise) args.push("--denoise");

  const res = await execCmd(cmd, args, { cwd: env.workdir });
  assertOk(res, "preprocess_audio");
  return { ok: true, out_path: input.out_path, stdout: res.stdout };
}

export async function transcribeAudio(input: z.infer<typeof ToolSchemas.transcribe_audio>) {
  const env = input.env;
  const { cmd, baseArgs } = cliCommand(env);

  const args = [...baseArgs, "transcribe", input.audio_path, "--out", input.out_path];
  if (input.bandpass) args.push("--bandpass");
  if (input.denoise) args.push("--denoise");

  const extraEnv: Record<string, string> = {};
  if (input.model) extraEnv["EMS_ASR_MODEL"] = input.model;

  const res = await execCmd(cmd, args, { cwd: env.workdir, env: extraEnv });
  assertOk(res, "transcribe_audio");

  return { ok: true, out_path: input.out_path, stdout: res.stdout };
}

export async function extractEntities(input: z.infer<typeof ToolSchemas.extract_entities>) {
  const env = input.env;
  const { cmd, baseArgs } = cliCommand(env);

  const args = [...baseArgs, "extract", input.transcript_json, "--out", input.out_path];
  const res = await execCmd(cmd, args, { cwd: env.workdir });
  assertOk(res, "extract_entities");

  return { ok: true, out_path: input.out_path, stdout: res.stdout };
}

export async function buildClaim(input: z.infer<typeof ToolSchemas.build_claim>) {
  const env = input.env;
  const { cmd, baseArgs } = cliCommand(env);

  const args = [...baseArgs, "build-claim", input.entities_json, "--out", input.out_path];
  const res = await execCmd(cmd, args, { cwd: env.workdir });
  assertOk(res, "build_claim");

  return { ok: true, out_path: input.out_path, stdout: res.stdout };
}

export async function runPipeline(input: z.infer<typeof ToolSchemas.run_pipeline>) {
  // End-to-end orchestration using the individual CLI commands
  const outTranscript = `${input.out_dir.replace(/\/$/, "")}/transcript.json`;
  const outEntities = `${input.out_dir.replace(/\/$/, "")}/entities.json`;
  const outClaim = `${input.out_dir.replace(/\/$/, "")}/claim.json`;

  await fs.mkdir(input.out_dir, { recursive: true });

  await transcribeAudio({
    audio_path: input.audio_path,
    out_path: outTranscript,
    bandpass: input.bandpass,
    denoise: input.denoise,
    model: input.model,
    env: input.env
  });

  await extractEntities({
    transcript_json: outTranscript,
    out_path: outEntities,
    env: input.env
  });

  await buildClaim({
    entities_json: outEntities,
    out_path: outClaim,
    env: input.env
  });

  // Return parsed claim for convenience
  const claim = await readJson(outClaim);
  return { ok: true, out_dir: input.out_dir, claim, paths: { outTranscript, outEntities, outClaim } };
}

export async function evalPred(input: z.infer<typeof ToolSchemas.eval>) {
  const env = input.env;
  const { cmd, baseArgs } = cliCommand(env);

  const args = [...baseArgs, "eval", "--gold", input.gold_path, "--pred", input.pred_path, "--out", input.out_path];
  const res = await execCmd(cmd, args, { cwd: env.workdir });
  assertOk(res, "eval");

  return { ok: true, out_path: input.out_path, stdout: res.stdout };
}