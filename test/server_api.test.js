import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

import {
  decodeServerMasks,
  getAvailableServerModels,
  getServerModelProvider,
} from '../js/server_api.js';

describe('server model metadata', () => {
  test('available model keys are normalized and deduplicated', () => {
    assert.deepEqual(
      getAvailableServerModels({ models: ['tiny', 'cellpose-cpsam-v2', 'tiny', null] }),
      ['tiny', 'cellpose-cpsam-v2'],
    );
    assert.deepEqual(getAvailableServerModels({}), []);
  });

  test('experimental backend provider is used for its models', () => {
    const info = {
      provider: 'CUDAExecutionProvider',
      experimentalModels: {
        cellpose: {
          models: ['cellpose-cpsam-v2'],
          provider: 'PyTorch CUDA',
        },
      },
    };

    assert.equal(getServerModelProvider(info, 'cellpose-cpsam-v2'), 'PyTorch CUDA');
    assert.equal(getServerModelProvider(info, 'tiny'), 'CUDAExecutionProvider');
  });
});

describe('decodeServerMasks', () => {
  test('start-length RLE masks are decoded into Uint8Array masks', () => {
    const decoded = decodeServerMasks({
      width: 5,
      height: 2,
      rawMasks: [
        {
          iou: 0.91,
          rle: { counts: [1, 3, 7, 2], encoding: 'start-length' },
        },
      ],
    });

    assert.equal(decoded.length, 1);
    assert.equal(decoded[0].iou, 0.91);
    assert.equal(decoded[0].width, 5);
    assert.equal(decoded[0].height, 2);
    assert.deepEqual([...decoded[0].mask], [
      0, 1, 1, 1, 0,
      0, 0, 1, 1, 0,
    ]);
  });

  test('snake_case response fields are accepted', () => {
    const decoded = decodeServerMasks({
      width: 3,
      height: 1,
      raw_masks: [
        {
          confidence: 0.5,
          rle: { counts: [0, 1, 2, 1] },
        },
      ],
    });

    assert.equal(decoded[0].iou, 0.5);
    assert.deepEqual([...decoded[0].mask], [1, 0, 1]);
  });
});
