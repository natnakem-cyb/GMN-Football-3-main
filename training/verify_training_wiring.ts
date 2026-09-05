/**
 * End-to-end training telemetry verification.
 *
 * Connects to the bridge server WebSocket, starts a short MAPPO training job,
 * and captures real parsed metrics flowing through the pipeline.
 *
 * This is a permanent, committed artifact — not a throwaway script.
 */

import WebSocket from 'ws';
import http from 'http';

const BRIDGE_HOST = process.env.GMN_BRIDGE_HOST || '127.0.0.1';
const BRIDGE_PORT = parseInt(process.env.GMN_BRIDGE_PORT || '5050', 10);
const WS_URL = `ws://${BRIDGE_HOST}:${BRIDGE_PORT}`;
const HTTP_BASE = `http://${BRIDGE_HOST}:${BRIDGE_PORT}`;

interface MetricsMessage {
  type: string;
  data?: any;
  snapshot?: any;
  payload?: any;
}

async function main() {
  console.log(`Connecting to bridge server at ${WS_URL}...`);
  
  const ws = new WebSocket(WS_URL);
  const messages: MetricsMessage[] = [];
  let metricsReceived = false;
  let trainingStarted = false;
  let trainingCompleted = false;

  const metricsPromise = new Promise<void>((resolve, reject) => {
    ws.on('open', () => {
      console.log('WebSocket connected');
      
      // Subscribe to metrics
      ws.send(JSON.stringify({ type: 'subscribe_metrics' }));
      console.log('Subscribed to metrics');
    });

    ws.on('message', (data: Buffer) => {
      try {
        const text = data.toString('utf8');
        const msg = JSON.parse(text);
        messages.push(msg);
        
        console.log(`[WS] ${msg.type}: ${JSON.stringify(msg.payload || msg.data || msg.snapshot || {}).slice(0, 200)}`);
        
        if (msg.type === 'training_metrics' || msg.type === 'TRAINING_METRICS' || msg.type === 'telemetry_metrics') {
          metricsReceived = true;
        }
        
        if (msg.type === 'TRAINING_STARTED' || msg.type === 'training_started') {
          trainingStarted = true;
          console.log('Training job started');
        }
        
        if (msg.type === 'TRAINING_COMPLETED' || msg.type === 'training_completed') {
          trainingCompleted = true;
          console.log('Training job completed');
          resolve();
        }
        
        if (msg.type === 'TRAINING_FAILED' || msg.type === 'training_failed') {
          reject(new Error(`Training failed: ${JSON.stringify(msg.data)}`));
        }
      } catch (err) {
        // Ignore non-JSON messages
      }
    });

    ws.on('error', (err) => {
      console.error('WebSocket error:', err);
      reject(err);
    });

    // Timeout after 120 seconds
    setTimeout(() => {
      if (!trainingCompleted) {
        resolve(); // Resolve anyway to see what we got
      }
    }, 120000);
  });

  // Start a short training job via HTTP API
  console.log('\nStarting short MAPPO training job (5000 steps)...');
  const postData = JSON.stringify({
    algorithm: 'mappo',
    scenario: 'academy_3_vs_1_with_keeper',
    timesteps: 5000,
  });

  const startPromise = new Promise<void>((resolve, reject) => {
    const req = http.request({
      hostname: BRIDGE_HOST,
      port: BRIDGE_PORT,
      path: '/api/training/start',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData),
      },
    }, (res) => {
      let body = '';
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        console.log(`Training start response: ${body}`);
        resolve();
      });
    });

    req.on('error', reject);
    req.write(postData);
    req.end();
  });

  await startPromise;
  await metricsPromise;

  ws.close();

  // Analyze results
  console.log('\n=== VERIFICATION RESULTS ===');
  console.log(`Training started: ${trainingStarted}`);
  console.log(`Training completed: ${trainingCompleted}`);
  console.log(`Metrics messages received: ${metricsReceived}`);
  console.log(`Total WebSocket messages: ${messages.length}`);

  // Find all training_metrics messages
  const metricsMessages = messages.filter(m => 
    m.type === 'training_metrics' || m.type === 'TRAINING_METRICS' || m.type === 'telemetry_metrics'
  );

  console.log(`\nTraining metrics messages: ${metricsMessages.length}`);

  if (metricsMessages.length > 0) {
    console.log('\nFirst 5 metrics messages:');
    for (let i = 0; i < Math.min(5, metricsMessages.length); i++) {
      const msg = metricsMessages[i];
      const data = msg.payload || msg.data || msg.snapshot || {};
      console.log(`  ${i + 1}. step=${data.step}, policyLoss=${data.policyLoss}, valueLoss=${data.valueLoss}, entropy=${data.entropy}, goalRate=${data.goalRate}`);
    }

    // Check that metrics actually change over time
    const steps = metricsMessages.map(m => {
      const data = m.payload || m.data || m.snapshot || {};
      return data.step;
    }).filter(s => s != null);

    const uniqueSteps = new Set(steps);
    console.log(`\nUnique step values: ${uniqueSteps.size}`);
    
    if (uniqueSteps.size > 1) {
      console.log('PASS: Metrics show multiple distinct step values (not frozen).');
    } else {
      console.log('WARN: All metrics have the same step value.');
    }

    // Check for fabricated fallback values
    const lastMsg = metricsMessages[metricsMessages.length - 1];
    const lastData = lastMsg.payload || lastMsg.data || lastMsg.snapshot || {};
    
    // Note: policyLoss may be null for MAPPO (not printed in stdout), which is correct.
    // The actual fabricated fallbacks were hardcoded numbers like -0.04, 1.8, 65.0.
    // learningRate=0.0003 is real data from MAPPO stdout "LR: 3e-4", not fabricated.
    const fabricatedValues: Record<string, number> = {
      policyLoss: -0.04,  // MAPPO may return null for this, which is correct
      valueLoss: 0.05,
      entropy: 1.8,
      approxKl: 0.01,
      clipFraction: 0.06,
      gradNorm: 0.1,
      rollingReward: 0.5,
      goalRate: 65.0,
    };

    let foundFabricated = false;
    for (const [key, fakeValue] of Object.entries(fabricatedValues)) {
      if (lastData[key] === fakeValue) {
        console.error(`FAIL: Found fabricated fallback for ${key}: ${fakeValue}`);
        foundFabricated = true;
      }
    }

    if (!foundFabricated) {
      console.log('PASS: No fabricated fallback values detected in final metrics.');
    }
  } else {
    console.log('WARN: No training_metrics messages received. Check bridge server and training job output.');
  }

  console.log('\n=== END OF VERIFICATION ===');
}

main().catch((err) => {
  console.error('Verification failed:', err);
  process.exit(1);
});
