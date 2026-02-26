import { spawn } from "node:child_process";

export type ExecResult = { code: number; stdout: string; stderr: string };

export function execCmd(
  cmd: string,
  args: string[],
  opts?: { cwd?: string; env?: Record<string, string> }
): Promise<ExecResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {
      stdio: ["ignore", "pipe", "pipe"],
      cwd: opts?.cwd,
      env: { ...process.env, ...(opts?.env ?? {}) }
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (d) => (stdout += d.toString("utf8")));
    child.stderr.on("data", (d) => (stderr += d.toString("utf8")));

    child.on("error", reject);
    child.on("close", (code) => resolve({ code: code ?? 1, stdout, stderr }));
  });
}

export function assertOk(res: ExecResult, context: string) {
  if (res.code !== 0) {
    const msg = [
      `${context} failed (exit=${res.code})`,
      res.stderr ? `stderr:\n${res.stderr}` : "",
      res.stdout ? `stdout:\n${res.stdout}` : ""
    ]
      .filter(Boolean)
      .join("\n\n");
    throw new Error(msg);
  }
}