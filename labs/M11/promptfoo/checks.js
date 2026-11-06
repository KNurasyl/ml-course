// JavaScript assertions for promptfoo (extension 1.2.0, T38, lesson M11-L06).
// validateFeedback repeats demo.validate_feedback of M09-L05 (labs/M09/demo.py) line by line;
// tests/test_promptfoo_eval.py runs both implementations on the same strings, edge cases included.
'use strict';

// Python's str.strip() whitespace. String.prototype.trim() differs: it keeps \x1c-\x1f and \x85
// but removes \ufeff, so trim() alone would accept a feedback that Python rejects as empty.
const PY_SPACE = /^[\t\n\v\f\r\x1c-\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/;
const INT_LITERAL = /^-?(0|[1-9][0-9]*)$/;

// JSON.parse turns 1 and 1.0 into the same number. Python keeps 1.0 a float and the contract rejects it,
// so the score literal is read from the reviver's source text (Node 22+).
function parse(text) {
  const sources = new Map();
  const value = JSON.parse(text, function (key, val, context) {
    if (key === 'score' && context && typeof context.source === 'string') sources.set(this, context.source);
    return val;
  });
  return { value, scoreSource: sources.get(value) };
}

function validateFeedback(text) {
  let parsed;
  try {
    parsed = parse(String(text));
  } catch {
    return { valid: false, reason: 'invalid_json' };
  }
  const { value, scoreSource } = parsed;
  const keys = value !== null && typeof value === 'object' && !Array.isArray(value) ? Object.keys(value).sort() : [];
  if (keys.length !== 2 || keys[0] !== 'feedback' || keys[1] !== 'score') return { valid: false, reason: 'wrong_keys' };
  if (typeof value.feedback !== 'string' || PY_SPACE.test(value.feedback)) return { valid: false, reason: 'empty_feedback' };
  const integer = Number.isInteger(value.score) && INT_LITERAL.test(scoreSource ?? '');
  if (!integer || value.score < 0 || value.score > 2) return { valid: false, reason: 'score_out_of_range' };
  return { valid: true, value };
}

// The M09-L05 contract: exactly feedback (non-empty string) and score (integer 0..2).
function contract(output) {
  const result = validateFeedback(output);
  return { pass: result.valid, score: result.valid ? 1 : 0, reason: result.valid ? 'contract ok' : result.reason };
}

// The score agrees with the author's gold label; an answer outside the contract has no score to compare.
function score(output, context) {
  const result = validateFeedback(output);
  if (!result.valid) return { pass: false, score: 0, reason: `no score: ${result.reason}` };
  const gold = Number(context.vars.gold);
  const pass = result.value.score === gold;
  return { pass, score: pass ? 1 : 0, reason: `score ${result.value.score}, gold ${gold}` };
}

module.exports = { contract, score, validateFeedback };
