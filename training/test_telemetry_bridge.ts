/**
 * GMN-Football-3 — Bridge Protocol & Telemetry Integration Test
 *
 * Spawns a real bridge server on an ephemeral port and verifies:
 *   - P0 #5: invalid single/multi-agent binary actions get a deterministic
 *     error frame (never silence)
 *   - P1 #8/#9: canonical telemetry protocol + explicit subscribe_training
 *   - P1 #10: hardware_stats / checkpoint_update / training_status reach a
 *     subscribed client
 *   - Contract: single-agent & multi-agent binary frame dimensions (127-dim)
 *   - P1 #13: reset responses are structurally valid JSON payloads
 */
import { spawn } from 'child_process';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';
import WebSocket from 'ws';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const PORT = 5099;
const BASE = `http://127.0.0.1:${PORT}`;
const OBS_DIM = 127;
const FRAME = 17 + OBS_DIM * 4;

let failures = 0;
function check(cond: boolean, label: string) {
  if (cond) console.log(`   OK  ${label}`);
  else {
    failures++;
    console.error(`   FAIL ${label}`);
  }
}

function waitForHealth(): Promise<void> {
  return new Promise((resolve, reject) => {
    let tries = 0;
    const poll = () => {
      http
        .get(`${BASE}/health`, (res) => {
          res.resume();
          if (res.statusCode === 200) resolve();
          else retry();
        })
        .on('error', retry);
    };
    const retry = () => {
      if (++tries > 50) reject(new Error('bridge did not start'));
      else setTimeout(poll, 300);
    };
    poll();
  });
}

function wsConnect(): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://127.0.0.1:${PORT}`);
    (ws as any)._queue = [];
    ws.on('message', (data: Buffer, isBinary: boolean) => {
      (ws as any)._queue.push({ msg: isBinary ? undefined : JSON.parse(data.toString('utf8')), buf: isBinary ? data : undefined });
    });
    ws.on('open', () => resolve(ws));
    ws.on('error', reject);
  });
}

function recvUntil(ws: WebSocket, pred: (m: any, buf?: Buffer) => boolean, timeoutMs = 10000): Promise<{ msg?: any; buf?: Buffer }> {
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error('recvUntil timeout')), timeoutMs);
    const tryQueue = () => {
      const q: any[] = (ws as any)._queue;
      while (q.length > 0) {
        const item = q.shift()!;
        if (pred(item.msg, item.buf)) {
          clearTimeout(to);
          resolve(item);
          return true;
        }
      }
      return false;
    };
    if (!tryQueue()) {
      const onMsg = () => {
        if (tryQueue()) ws.off('message', onMsg);
      };
      ws.on('message', onMsg);
    }
  });
}

async function main() {
  console.log('====================================================');
  console.log('GMN-FOOTBALL-3 - BRIDGE PROTOCOL & TELEMETRY TEST');
  console.log('====================================================');

  const bridge = spawn('npx', ['tsx', path.join(ROOT, 'training', 'bridge_server.ts')], {
    env: { ...process.env, GMN_BRIDGE_PORT: String(PORT) },
    shell: process.platform.startsWith('win'),
    stdio: 'ignore',
  });
  try {
    await waitForHealth();
    console.log('   OK  bridge started');

    // --- P1 #9/#10: explicit subscribe_training -> canonical telemetry ---
    const dash = await wsConnect();
    dash.send(JSON.stringify({ type: 'subscribe_training' }));
    const status = await recvUntil(dash, (m) => m && m.type === 'training_status');
    check(
      status.msg && status.msg.type === 'training_status' && status.msg.data && typeof status.msg.data.isRunning === 'boolean',
      'subscribe_training answered with canonical {type:"training_status", data:{...}}'
    );
    const ck = await recvUntil(dash, (m) => m && m.type === 'checkpoint_update');
    check(Array.isArray(ck.msg.data.checkpoints), 'checkpoint_update reaches the subscribed client (data.checkpoints array)');

    const hw = await recvUntil(dash, (m) => m && m.type === 'hardware_stats', 12000);
    check(
      hw.msg && typeof hw.msg.data.cpuPercent === 'number' && typeof hw.msg.data.stepsPerSec === 'number',
      'hardware_stats (CPU / SPS) reaches the subscribed client'
    );

    // --- Contract: reset response is structurally valid, 127-dim ---
    dash.send(JSON.stringify({ type: 'reset', scenario: 'academy_3_vs_1_with_keeper', seed: 123 }));
    const reset = await recvUntil(dash, (m) => m && (m.observation || m.observations));
    const obs = reset.msg.observation || reset.msg.observations[0];
    check(Array.isArray(obs) && obs.length === OBS_DIM, 'reset response observation is 127-dim');
    const leftAgents = (reset.msg.observations || [obs]).length;

    // --- Contract: valid single-agent step frame ---
    dash.send(Buffer.from([0]), { binary: true });
    const okStep = await recvUntil(dash, (_m, b) => !!b);
    check(okStep.buf!.length === FRAME, `valid single-agent step returns ${FRAME}-byte frame`);

    // --- P0 #5: invalid single-agent action -> deterministic error frame ---
    dash.send(Buffer.from([200]), { binary: true });
    const err1 = await recvUntil(dash, (_m, b) => !!b);
    check(
      err1.buf!.length === FRAME && err1.buf!.readFloatLE(0) === -999.0 && err1.buf!.readUInt8(4) === 1,
      'invalid single-agent action returns error frame (length preserved, sentinel -999, terminated=1)'
    );

    // --- Contract: valid multi-agent step frame (N left agents) ---
    dash.send(Buffer.alloc(leftAgents, 0), { binary: true });
    const multiOk = await recvUntil(dash, (_m, b) => !!b);
    check(
      multiOk.buf!.length === 17 + OBS_DIM * 4 * leftAgents,
      `multi-agent step returns exactly one frame for ${leftAgents} agents (${17 + OBS_DIM * 4 * leftAgents} bytes)`
    );

    // --- P0 #5: invalid multi-agent action -> error frame sized to N ---
    dash.send(Buffer.alloc(leftAgents, 250), { binary: true });
    const errN = await recvUntil(dash, (_m, b) => !!b);
    check(
      errN.buf!.length === 17 + OBS_DIM * 4 * leftAgents && errN.buf!.readFloatLE(0) === -999.0,
      'invalid multi-agent action returns N-sized error frame (no silence, no hang)'
    );

    // --- Malformed request -> canonical error message ---
    dash.send('this is not json');
    const errJson = await recvUntil(dash, (m) => m && m.type === 'error');
    check(typeof errJson.msg.data?.message === 'string', 'malformed request returns canonical {type:"error"} message');

    dash.close();
    console.log('   (python-side validation covered by training/test_gym_safety.py)');
  } finally {
    // Windows: npx spawns a process tree — kill the whole tree, not just the shell.
    if (bridge.pid) {
      if (process.platform.startsWith('win')) {
        spawn('taskkill', ['/pid', String(bridge.pid), '/T', '/F'], { stdio: 'ignore' });
      } else {
        bridge.kill();
      }
    }
  }

  console.log('\n====================================================');
  if (failures === 0) console.log('OK  ALL BRIDGE PROTOCOL & TELEMETRY CHECKS PASSED');
  else {
    console.error(`FAIL ${failures} BRIDGE CHECK(S) FAILED`);
    process.exit(1);
  }
  console.log('====================================================');
}

main().catch((e) => {
  console.error('BRIDGE TEST ERROR:', e);
  process.exit(1);
});
