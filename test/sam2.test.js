import { test } from 'node:test';
import assert from 'node:assert/strict';

import { SAM2 } from '../js/sam2.js';


test('SAM2 can initialize without eagerly loading ONNX Runtime Web', async () => {
  const sam2 = new SAM2();

  assert.equal(sam2.encoderSession, null);
  assert.equal(sam2.decoderSession, null);
  assert.equal(await sam2.detectExecutionProvider(), 'wasm');
});
