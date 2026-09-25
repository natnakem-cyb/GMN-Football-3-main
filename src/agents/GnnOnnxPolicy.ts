import * as ort from 'onnxruntime-web';
import { ACTION_SPACE_SIZE } from '../engine/Contract';

/** Tensorized v3 graph contract accepted by training/gnn_onnx.py exports. */
export interface GnnOnnxGraphInput {
  nodeFeatures: Float32Array;
  nodeCount: number;
  nodeType: BigInt64Array;
  edgeIndex: BigInt64Array;
  edgeCount: number;
  edgeType: BigInt64Array;
  edgeFeatures: Float32Array;
  nodeMask: Float32Array;
  graphContext: Float32Array;
  agentNodeIndex: number;
}

const REQUIRED_INPUTS = ['node_features', 'agent_node_index'] as const;

/**
 * Browser ONNX runner for exported GNN actors.
 *
 * The caller supplies tensorized graphs. This class does not build graphs from
 * GameEngine state; the matching canonical preprocessing currently lives in
 * training.gnn_graph_builder and training.gnn_graph_to_tensor.
 */
export class GnnOnnxPolicy {
  private constructor(private readonly session: ort.InferenceSession) {}

  static async create(
    modelSource: string | ArrayBuffer | Uint8Array,
  ): Promise<GnnOnnxPolicy> {
    if (typeof ort !== 'undefined' && ort.env?.wasm) {
      ort.env.wasm.wasmPaths = '/onnx/';
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
    }
    const session =
      typeof modelSource === 'string'
        ? await ort.InferenceSession.create(modelSource)
        : modelSource instanceof Uint8Array
          ? await ort.InferenceSession.create(modelSource)
          : await ort.InferenceSession.create(modelSource);
    const missing = REQUIRED_INPUTS.filter((name) => !session.inputNames.includes(name));
    if (missing.length) {
      await session.release();
      throw new Error(`GNN ONNX model is missing graph inputs: ${missing.join(', ')}`);
    }
    return new GnnOnnxPolicy(session);
  }

  async predictLogits(graph: GnnOnnxGraphInput): Promise<Float32Array> {
    this.validateGraph(graph);
    const candidateFeeds: Record<string, ort.Tensor> = {
      node_features: new ort.Tensor('float32', graph.nodeFeatures, [graph.nodeCount, 32]),
      edge_index: new ort.Tensor('int64', graph.edgeIndex, [2, graph.edgeCount]),
      edge_features: new ort.Tensor('float32', graph.edgeFeatures, [graph.edgeCount, 10]),
      node_mask: new ort.Tensor('float32', graph.nodeMask, [graph.nodeCount]),
      graph_context: new ort.Tensor('float32', graph.graphContext, [8]),
      agent_node_index: new ort.Tensor(
        'int64',
        BigInt64Array.of(BigInt(graph.agentNodeIndex)),
        [1],
      ),
    };
    const feeds: Record<string, ort.Tensor> = {};
    for (const name of this.session.inputNames) {
      const tensor = candidateFeeds[name];
      if (!tensor) throw new Error(`Unsupported GNN ONNX input: ${name}`);
      feeds[name] = tensor;
    }
    const results = await this.session.run(feeds);
    const output = results.action_logits || Object.values(results)[0];
    if (!output?.data || output.data.length !== ACTION_SPACE_SIZE) {
      throw new Error('GNN ONNX model returned invalid action logits.');
    }
    return output.data as Float32Array;
  }

  async predictAction(
    graph: GnnOnnxGraphInput,
    actionMask: ArrayLike<number>,
  ): Promise<number> {
    if (actionMask.length !== ACTION_SPACE_SIZE) {
      throw new Error(`Expected ${ACTION_SPACE_SIZE} legal-action flags.`);
    }
    const logits = await this.predictLogits(graph);
    let bestAction = -1;
    let bestLogit = -Infinity;
    for (let action = 0; action < ACTION_SPACE_SIZE; action += 1) {
      if (actionMask[action] && logits[action] > bestLogit) {
        bestLogit = logits[action];
        bestAction = action;
      }
    }
    if (bestAction < 0) throw new Error('Action mask contains no legal actions.');
    return bestAction;
  }

  async release(): Promise<void> {
    await this.session.release();
  }

  private validateGraph(graph: GnnOnnxGraphInput): void {
    if (!Number.isInteger(graph.nodeCount) || graph.nodeCount < 1) {
      throw new Error('GNN graph must contain at least one node.');
    }
    if (!Number.isInteger(graph.edgeCount) || graph.edgeCount < 0) {
      throw new Error('GNN edge count must be a non-negative integer.');
    }
    const expected = [
      [graph.nodeFeatures.length, graph.nodeCount * 32, 'node features'],
      [graph.edgeIndex.length, graph.edgeCount * 2, 'edge indices'],
      [graph.edgeFeatures.length, graph.edgeCount * 10, 'edge features'],
      [graph.nodeMask.length, graph.nodeCount, 'node mask'],
      [graph.graphContext.length, 8, 'graph context'],
    ] as const;
    for (const [actual, wanted, name] of expected) {
      if (actual !== wanted) {
        throw new Error(`GNN ${name} length ${actual} does not match expected ${wanted}.`);
      }
    }
    if (
      !Number.isInteger(graph.agentNodeIndex) ||
      graph.agentNodeIndex < 0 ||
      graph.agentNodeIndex >= graph.nodeCount
    ) {
      throw new Error('GNN controlled-agent node index is out of range.');
    }
  }
}
