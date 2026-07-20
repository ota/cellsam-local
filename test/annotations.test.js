import { describe, test } from 'node:test';
import assert from 'node:assert/strict';

import {
  DRAFT_KIND,
  GROUND_TRUTH_KIND,
  createDraftAnnotation,
  decodeMaskRle,
  documentFromEditable,
  editableObjectsFromDocument,
  encodeMaskRle,
  validateAnnotationDocument,
} from '../js/annotations.js';

describe('annotation RLE', () => {
  test('round-trips a row-major binary mask', () => {
    const mask = Uint8Array.from([0, 1, 1, 0, 0, 1, 0, 1]);
    const rle = encodeMaskRle(mask);

    assert.deepEqual(rle, {
      encoding: 'start-length-row-major',
      counts: [1, 2, 5, 1, 7, 1],
    });
    assert.deepEqual(decodeMaskRle(rle, 4, 2), mask);
  });

  test('rejects overlapping and out-of-bounds spans', () => {
    assert.throws(
      () => decodeMaskRle({ counts: [1, 4, 3, 2] }, 3, 2),
      /overlap/,
    );
    assert.throws(
      () => decodeMaskRle({ counts: [5, 2] }, 3, 2),
      /exceed/,
    );
  });
});

describe('annotation documents', () => {
  test('creates a draft and preserves rejected objects and notes', () => {
    const document = createDraftAnnotation({
      image: { fileName: 'sample.png', width: 3, height: 2 },
      source: { modelKey: 'tiny', backend: 'local' },
      objects: [
        {
          objId: 4,
          mask: Uint8Array.from([0, 1, 1, 0, 0, 0]),
          confidence: 0.9,
          excluded: true,
          notes: 'edge',
        },
      ],
    });

    assert.equal(document.kind, DRAFT_KIND);
    assert.equal(document.objects[0].id, 4);
    assert.equal(document.objects[0].status, 'rejected');
    assert.equal(document.objects[0].notes, 'edge');
    assert.doesNotThrow(() => validateAnnotationDocument(document));
  });

  test('exports edited masks as approved ground truth', () => {
    const draft = createDraftAnnotation({
      image: { fileName: 'sample.png', width: 2, height: 2 },
      source: { modelKey: 'microsam-vit-b-lm' },
      objects: [{ id: 1, mask: Uint8Array.from([1, 0, 0, 0]) }],
    });
    const editable = editableObjectsFromDocument(draft);
    editable[0].mask[3] = 1;
    editable[0].status = 'accepted';

    const groundTruth = documentFromEditable({
      baseDocument: draft,
      objects: editable,
      kind: GROUND_TRUTH_KIND,
    });

    assert.equal(groundTruth.kind, GROUND_TRUTH_KIND);
    assert.equal(groundTruth.review.status, 'approved');
    assert.equal(groundTruth.objects[0].areaPixels, 2);
    assert.deepEqual(
      decodeMaskRle(groundTruth.objects[0].rle, 2, 2),
      Uint8Array.from([1, 0, 0, 1]),
    );
  });
});
