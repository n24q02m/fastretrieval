# CHANGELOG

<!-- version list -->

## v1.13.0 (2026-08-08)


## v1.13.0-beta.1 (2026-08-08)

### Bug Fixes

- Accept empty and whitespace-only text in check_input_length
  ([#966](https://github.com/n24q02m/qwen3-embed/pull/966),
  [`35a525a`](https://github.com/n24q02m/qwen3-embed/commit/35a525a02a0c0c4c9f03b0ae5cc48493946327cd))

- Add regex search fast path to input sanitization
  ([`cc6a184`](https://github.com/n24q02m/qwen3-embed/commit/cc6a1840c323023a956ec127ec729c435f52dd35))

- Adopt better-semantic-release for built-in release guards
  ([`d0427d0`](https://github.com/n24q02m/qwen3-embed/commit/d0427d03f450b97c3ffa806987d050d96c6e7ec5))

- Assert the link refusal without removing tarfile.data_filter
  ([#958](https://github.com/n24q02m/qwen3-embed/pull/958),
  [`33ed74e`](https://github.com/n24q02m/qwen3-embed/commit/33ed74e278aa935c13eedd805c43cf94cad79344))

- Clamp embedding batch size to the ONNX graph batch axis
  ([#970](https://github.com/n24q02m/qwen3-embed/pull/970),
  [`d9dc0be`](https://github.com/n24q02m/qwen3-embed/commit/d9dc0be2efd473d939dee650372b1b2fc06542a6))

- Clear the GCS staging path whatever type it is
  ([#963](https://github.com/n24q02m/qwen3-embed/pull/963),
  [`f426f9c`](https://github.com/n24q02m/qwen3-embed/commit/f426f9c64128bb17e9bb33fbda9147ba361bb79e))

- Correct mention gate expression (balanced parens + precedence)
  ([#914](https://github.com/n24q02m/qwen3-embed/pull/914),
  [`8764f0c`](https://github.com/n24q02m/qwen3-embed/commit/8764f0c4aae1be8c19b7a07506e2e036bdf0b4bd))

- Document backend-only repository limitation
  ([`26e07a5`](https://github.com/n24q02m/qwen3-embed/commit/26e07a5b571f24ba406ac8614cf1ed985f587bda))

- Drop code_quality from the main ruleset ([#935](https://github.com/n24q02m/qwen3-embed/pull/935),
  [`a9c7489`](https://github.com/n24q02m/qwen3-embed/commit/a9c74896ba586bf87599037b3e3f5e03280f34d8))

- Drop rangeStrategy from update-type package rules
  ([#929](https://github.com/n24q02m/qwen3-embed/pull/929),
  [`3426710`](https://github.com/n24q02m/qwen3-embed/commit/342671019fc8caa926224ba091f6bdd1f04b28ba))

- Enforce fix(deps) semantic commit prefix in renovate config
  ([`4c7e45f`](https://github.com/n24q02m/qwen3-embed/commit/4c7e45f34d154b2a6ac8d14dbb2f4a2db6ad3689))

- Fail the release when the computed version already exists on PyPI
  ([#897](https://github.com/n24q02m/qwen3-embed/pull/897),
  [`59b22ed`](https://github.com/n24q02m/qwen3-embed/commit/59b22ed7b046aacf5a87c53ba5a30c818eb02338))

- Format README code blocks the way ruff 0.16 wants them
  ([#964](https://github.com/n24q02m/qwen3-embed/pull/964),
  [`20c8adc`](https://github.com/n24q02m/qwen3-embed/commit/20c8adc6652f47f316a7ccf4fabc18f7867c4b94))

- Gate oc mention job on comment author write access
  ([#914](https://github.com/n24q02m/qwen3-embed/pull/914),
  [`8764f0c`](https://github.com/n24q02m/qwen3-embed/commit/8764f0c4aae1be8c19b7a07506e2e036bdf0b4bd))

- Let a fixed PR title clear the title gate
  ([#959](https://github.com/n24q02m/qwen3-embed/pull/959),
  [`74239b3`](https://github.com/n24q02m/qwen3-embed/commit/74239b33cf7dba2258fb606358175a1b7236b591))

- Let Renovate lowercase its own PR titles ([#965](https://github.com/n24q02m/qwen3-embed/pull/965),
  [`b1b00a9`](https://github.com/n24q02m/qwen3-embed/commit/b1b00a98f4a6aed5a86cf93a9d4720c105bc9226))

- Let the integration cache bootstrap itself and report its own progress
  ([#972](https://github.com/n24q02m/qwen3-embed/pull/972),
  [`15695bb`](https://github.com/n24q02m/qwen3-embed/commit/15695bb9990d512c1a5db1e1221f7282022f4925))

- Make a hanging integration test name itself instead of dying silently
  ([#972](https://github.com/n24q02m/qwen3-embed/pull/972),
  [`15695bb`](https://github.com/n24q02m/qwen3-embed/commit/15695bb9990d512c1a5db1e1221f7282022f4925))

- Make renovate automerge effective (isolated groups, digest+lockfile automerge, 7-day cooldown)
  ([`722455d`](https://github.com/n24q02m/qwen3-embed/commit/722455d8caaeff01524b8264034fbe2801e034ad))

- Optimize FP16 to FP32 Casting
  ([`d26a5fe`](https://github.com/n24q02m/qwen3-embed/commit/d26a5fe79f5643354c3951cad65e88e8b72166fb))

- Optimize regex substitution loops with search fast-path in gguf cross encoder
  ([`2278801`](https://github.com/n24q02m/qwen3-embed/commit/2278801b112b13afcd79e3bd1c80405f16494110))

- Pin GitHub Action references to commit SHAs
  ([#925](https://github.com/n24q02m/qwen3-embed/pull/925),
  [`94ccbdf`](https://github.com/n24q02m/qwen3-embed/commit/94ccbdfb647d6bb9dde350f7f4fd370978ad7bf7))

- Pin rangeStrategy on delayed packageRules to fix renovate artifacts failure
  ([#920](https://github.com/n24q02m/qwen3-embed/pull/920),
  [`3b51822`](https://github.com/n24q02m/qwen3-embed/commit/3b51822803ac32ffbadb3d3a274407851c8bbd32))

- Point the custom-reranker test at a model its code path can score
  ([#972](https://github.com/n24q02m/qwen3-embed/pull/972),
  [`15695bb`](https://github.com/n24q02m/qwen3-embed/commit/15695bb9990d512c1a5db1e1221f7282022f4925))

- Refuse tar symlinks that escape the model cache
  ([#958](https://github.com/n24q02m/qwen3-embed/pull/958),
  [`33ed74e`](https://github.com/n24q02m/qwen3-embed/commit/33ed74e278aa935c13eedd805c43cf94cad79344))

- Regenerate uv.lock after rebasing dependency bumps onto main
  ([#930](https://github.com/n24q02m/qwen3-embed/pull/930),
  [`f1b419a`](https://github.com/n24q02m/qwen3-embed/commit/f1b419aebe516927874bb587bbe24cf0ce1828c3))

- Replace noise-dominated retrieval assertion in test_full_pipeline
  ([#967](https://github.com/n24q02m/qwen3-embed/pull/967),
  [`2894b2b`](https://github.com/n24q02m/qwen3-embed/commit/2894b2b17bf50365bea43f2249a52d06684128c2))

- Restore Python 3.11 support and stop ty failing CI on warnings
  ([#898](https://github.com/n24q02m/qwen3-embed/pull/898),
  [`fda5944`](https://github.com/n24q02m/qwen3-embed/commit/fda594483c010372bd67f57aa34e84f0f869e20c))

- Reuse mask sum in mean pooling
  ([`261de55`](https://github.com/n24q02m/qwen3-embed/commit/261de55335d936984a5963c8b60ad50e606b33b5))

- Run opencode bot on hosted runners ([#913](https://github.com/n24q02m/qwen3-embed/pull/913),
  [`f4e300d`](https://github.com/n24q02m/qwen3-embed/commit/f4e300d067a6ab08ed79e5c7084c20d524089603))

- Scope the MAX_INPUT_LENGTH override to this test module
  ([#971](https://github.com/n24q02m/qwen3-embed/pull/971),
  [`25b411f`](https://github.com/n24q02m/qwen3-embed/commit/25b411f6d324d4a84c47a2ab3debc6e996540969))

- Stop injecting a fake llama_cpp into sys.modules for the whole session
  ([#971](https://github.com/n24q02m/qwen3-embed/pull/971),
  [`25b411f`](https://github.com/n24q02m/qwen3-embed/commit/25b411f6d324d4a84c47a2ab3debc6e996540969))

- Stop the worker pool deadlocking when its consumer walks away
  ([#972](https://github.com/n24q02m/qwen3-embed/pull/972),
  [`15695bb`](https://github.com/n24q02m/qwen3-embed/commit/15695bb9990d512c1a5db1e1221f7282022f4925))

- Stop two test modules from mutating global state at import time
  ([#971](https://github.com/n24q02m/qwen3-embed/pull/971),
  [`25b411f`](https://github.com/n24q02m/qwen3-embed/commit/25b411f6d324d4a84c47a2ab3debc6e996540969))

- Sum integer masks before float cast in mean pooling
  ([`00ef2aa`](https://github.com/n24q02m/qwen3-embed/commit/00ef2aaeba53cca2ccbadba9e0c81a0271f73164))

- Trigger the integration suite on every test file that holds one
  ([#972](https://github.com/n24q02m/qwen3-embed/pull/972),
  [`15695bb`](https://github.com/n24q02m/qwen3-embed/commit/15695bb9990d512c1a5db1e1221f7282022f4925))

- Use fixed-point string replacement for input sanitization
  ([`6d39950`](https://github.com/n24q02m/qwen3-embed/commit/6d399504e25c854b1b266648614ee509a99b45c2))

- **deps**: Lock file maintenance ([#962](https://github.com/n24q02m/qwen3-embed/pull/962),
  [`7a19f2d`](https://github.com/n24q02m/qwen3-embed/commit/7a19f2dcbb0aa91f7cacebdb45e3140cee8c9de6))

- **deps**: Lock file maintenance ([#918](https://github.com/n24q02m/qwen3-embed/pull/918),
  [`766c765`](https://github.com/n24q02m/qwen3-embed/commit/766c765841e559d2bcf78642b03adad7a278ccca))

- **deps**: Lock file maintenance ([#915](https://github.com/n24q02m/qwen3-embed/pull/915),
  [`c7991f7`](https://github.com/n24q02m/qwen3-embed/commit/c7991f7f492ba638a2f1791fe2672aba4bd6d3f6))

- **deps**: Lock file maintenance ([#911](https://github.com/n24q02m/qwen3-embed/pull/911),
  [`f6b32b5`](https://github.com/n24q02m/qwen3-embed/commit/f6b32b56eb32420fa348fdf6a1397dd0ab93b5bf))

- **deps**: Lock file maintenance ([#908](https://github.com/n24q02m/qwen3-embed/pull/908),
  [`21adfd0`](https://github.com/n24q02m/qwen3-embed/commit/21adfd086e3ad3a9955de4c06efd7930fb76b7b1))

- **deps**: Update actions/checkout action to v7
  ([#907](https://github.com/n24q02m/qwen3-embed/pull/907),
  [`46a7042`](https://github.com/n24q02m/qwen3-embed/commit/46a704246b33331f6870b90b12916f03e4a3debf))

- **deps**: Update actions/setup-python action to v7
  ([#931](https://github.com/n24q02m/qwen3-embed/pull/931),
  [`24e2260`](https://github.com/n24q02m/qwen3-embed/commit/24e2260b0ceacd965fe53099ea068a9430772dc0))

- **deps**: Update astral-sh/setup-uv action to v8.3.2
  ([`8101060`](https://github.com/n24q02m/qwen3-embed/commit/8101060a40f6b46effd7d8c6f998795c275dd79c))

- **deps**: Update astral-sh/setup-uv action to v9
  ([#932](https://github.com/n24q02m/qwen3-embed/pull/932),
  [`a2d7fec`](https://github.com/n24q02m/qwen3-embed/commit/a2d7fecd9465fe111be52865adc08d1b6ad8b71b))

- **deps**: Update minor dependencies ([#930](https://github.com/n24q02m/qwen3-embed/pull/930),
  [`f1b419a`](https://github.com/n24q02m/qwen3-embed/commit/f1b419aebe516927874bb587bbe24cf0ce1828c3))

- **deps**: Update patch dependencies ([#901](https://github.com/n24q02m/qwen3-embed/pull/901),
  [`6120505`](https://github.com/n24q02m/qwen3-embed/commit/6120505a95b2d75cd2123b8b276dd2a59c1f0829))

- **deps**: Update ruff to >=0.16.1 ([#961](https://github.com/n24q02m/qwen3-embed/pull/961),
  [`8d0e329`](https://github.com/n24q02m/qwen3-embed/commit/8d0e32996f4d5d9781a4a1445bec2a4d6a6ede74))

### Features

- Add bot PR governance to this repo ([#960](https://github.com/n24q02m/qwen3-embed/pull/960),
  [`9aff668`](https://github.com/n24q02m/qwen3-embed/commit/9aff668c002a976faf91ebdb9634a7cf7068b2d6))

- Add opencode github agent (responds to /oc)
  ([`364704e`](https://github.com/n24q02m/qwen3-embed/commit/364704e6b82aba2240e77d166b5882591d5747ac))

- Add PR-title conventional-commit gate + no-bump release warning
  ([#922](https://github.com/n24q02m/qwen3-embed/pull/922),
  [`aa47bfc`](https://github.com/n24q02m/qwen3-embed/commit/aa47bfcb815e297120f3242fde1d243c55e01231))

- Add review-learnings store the automated reviewer must obey
  ([`e3201f1`](https://github.com/n24q02m/qwen3-embed/commit/e3201f1b7f2e3e1c11464c62c717376a2c640bd4))

- Auto-respond only to issues and PRs opened by outside people
  ([`3a90360`](https://github.com/n24q02m/qwen3-embed/commit/3a903600a2f5dc5cf5ff6029df895d0614eafd1d))

- Reviewer must obey .github/review-learnings.md
  ([`cfbdd93`](https://github.com/n24q02m/qwen3-embed/commit/cfbdd9363a3ec0a5f96db7ae81d55d16831560f9))

- Run the integration suite in CI ([#972](https://github.com/n24q02m/qwen3-embed/pull/972),
  [`15695bb`](https://github.com/n24q02m/qwen3-embed/commit/15695bb9990d512c1a5db1e1221f7282022f4925))

- Sync cross-promo section ([#938](https://github.com/n24q02m/qwen3-embed/pull/938),
  [`f693e9f`](https://github.com/n24q02m/qwen3-embed/commit/f693e9f083820c97234de3b0842b0c91c44a5d14))

### Performance Improvements

- Optimize mean pooling sum reduction by reusing float cast mask
  ([`261de55`](https://github.com/n24q02m/qwen3-embed/commit/261de55335d936984a5963c8b60ad50e606b33b5))


## v1.12.1 (2026-07-05)


## v1.12.1-beta.1 (2026-07-05)

### Bug Fixes

- Fast-path scalar sigmoid with math.exp for batch size 1
  ([`53744a1`](https://github.com/n24q02m/qwen3-embed/commit/53744a11eba280dd5ef494eb23e58e744a977530))

- Sum integer attention mask before float cast in mean pooling
  ([`6702cd4`](https://github.com/n24q02m/qwen3-embed/commit/6702cd45ed04ba63f250a0287de49a25552ab000))

- Validate GCS download host via parsed.hostname to prevent SSRF
  ([`8d1c946`](https://github.com/n24q02m/qwen3-embed/commit/8d1c946414083fbac3790405ab4a45afeecaea43))

- **deps**: Lock file maintenance
  ([`b751f96`](https://github.com/n24q02m/qwen3-embed/commit/b751f96d28a9b6128e62f2d5a88c669766e93d95))

- **deps**: Update non-major dependencies
  ([`28697e2`](https://github.com/n24q02m/qwen3-embed/commit/28697e23fc92c796500944d3879b000c3816357f))


## v1.12.0 (2026-07-01)

### Bug Fixes

- Add check_worker_health test
  ([`dccf2f0`](https://github.com/n24q02m/qwen3-embed/commit/dccf2f0eef45c6f859f41da79077f2db35d2f840))

- Add cumulative and fallback-path tar bomb tests for decompress_to_cache
  ([`8d99154`](https://github.com/n24q02m/qwen3-embed/commit/8d99154c3d2d4e6581ddb6cc7ef16567deed4b63))

- Add edge-case tests for iter_batch (too-large size, empty iterable, zero-size non-consumption)
  ([`18c6666`](https://github.com/n24q02m/qwen3-embed/commit/18c6666a481ee30b48f4eecfdd55a4ce610cb488))

- Add error test for _validate_tar_member path traversal
  ([`5589597`](https://github.com/n24q02m/qwen3-embed/commit/558959743f88edecc319d0c3a16625278a97a734))

- Add error-path test for check_input_length
  ([`715fb16`](https://github.com/n24q02m/qwen3-embed/commit/715fb16ca4a212c79c1609d7b57b1e733f3cae8f))

- Add tests for regression test for over-strict bug
  ([`f4158f2`](https://github.com/n24q02m/qwen3-embed/commit/f4158f294b8a54616920cf95b9dfc29a542584c8))

- Add tests for too many arguments: make_model_description
  ([`d46931c`](https://github.com/n24q02m/qwen3-embed/commit/d46931c9d6925887e7184288154f3b183204a507))

- Add tests for untested error path in _save_file_metadata
  ([`2b8a50b`](https://github.com/n24q02m/qwen3-embed/commit/2b8a50bb9865dbf7f47d645c695158bc7e9e13a1))

- Correct doc rot in README intro and stale embedding docstrings
  ([#800](https://github.com/n24q02m/qwen3-embed/pull/800),
  [`29c95ea`](https://github.com/n24q02m/qwen3-embed/commit/29c95eab8edfef0364a0b2a40dffa3215cf38082))

- Correct last_token_pool for mixed-padding, empty, and all-zero attention masks
  ([`07f19ee`](https://github.com/n24q02m/qwen3-embed/commit/07f19eee2653273aa6517001614c640bc604f6e0))

- Cover export_to_onnx
  ([`2f434c9`](https://github.com/n24q02m/qwen3-embed/commit/2f434c919f22df796ad440bf1bc007913a1f0283))

- Cover internal worker loop and cleanup in parallel_processor
  ([`413fbe5`](https://github.com/n24q02m/qwen3-embed/commit/413fbe5b5b24a439a43137e379de4c293599ed58))

- Cover join_or_terminate states
  ([`5b186a6`](https://github.com/n24q02m/qwen3-embed/commit/5b186a67251ac239d19b1e0357a58cba52783fdd))

- Cover onnx_text_model
  ([`0a07dad`](https://github.com/n24q02m/qwen3-embed/commit/0a07dada871ec7371009985ea5cc0554d606b044))

- Drop unused llama_cpp import in GGUF cross-encoder dependency check
  ([`dfa6e97`](https://github.com/n24q02m/qwen3-embed/commit/dfa6e9716b984779f939c6aa2e70cce3513d51f2))

- Drop unused OnnxProvider/PathInput re-export from common package
  ([`c6eb038`](https://github.com/n24q02m/qwen3-embed/commit/c6eb038755cbe3e21f8a7c34914de1e2d27dcd91))

- Extract yes/no token logits without full-vocab slice
  ([`b862da4`](https://github.com/n24q02m/qwen3-embed/commit/b862da40d5d17cfde8b24892e2c0743f43cff868))

- Guard _cleanup_worker leak on queue.close error + cover
  ([`c12e397`](https://github.com/n24q02m/qwen3-embed/commit/c12e39738522efc9405758a8cf38c0e745ec9903))

- Import TextCrossEncoder from source module and drop redundant re-export
  ([`fc4f490`](https://github.com/n24q02m/qwen3-embed/commit/fc4f4904176ae6e0e0d4ee3514f87b5ed85d12c2))

- Mismatched dictionary key lookup for repo files
  ([`447d5e6`](https://github.com/n24q02m/qwen3-embed/commit/447d5e665fbbd5e151606cc76ec0e498c3da4158))

- Missing cache for supported models in TextEmbedding
  ([`788e65d`](https://github.com/n24q02m/qwen3-embed/commit/788e65dbb480b7a6827014d5638f06d3988b81ce))

- Missing edge case: is_safe_path with relative paths
  ([`0a137e4`](https://github.com/n24q02m/qwen3-embed/commit/0a137e42fe35825d0b5bfb3bff92bb110ffec90a))

- Missing edge case: semi_ordered_map with empty stream
  ([`30222d1`](https://github.com/n24q02m/qwen3-embed/commit/30222d1192e8668c1c3c41ce385d986da46f5d65))

- Missing error test for parallel worker cleanup
  ([`5cec788`](https://github.com/n24q02m/qwen3-embed/commit/5cec788104a0cae34f5d2d5fddec5cc1b4c7729c))

- Missing tests for custom_model.py
  ([`fc2d6af`](https://github.com/n24q02m/qwen3-embed/commit/fc2d6afb2eff92a9344a0ea9b37136e6a99e5406))

- Missing tests for export.py
  ([`b7cc5c0`](https://github.com/n24q02m/qwen3-embed/commit/b7cc5c0eb548f9ab68c1e7e14fd96877974bbad5))

- O(1) last_token_pool index lookup
  ([`8224a13`](https://github.com/n24q02m/qwen3-embed/commit/8224a13840be0fb3d568c1b7e498a36d24b2dabf))

- Pin numpy <2.5 to keep Python 3.11 support
  ([`4ad79c3`](https://github.com/n24q02m/qwen3-embed/commit/4ad79c3f2c946889d1c8aaf3a2faf72f39ce461c))

- Propagate custom reranker registry to spawned workers
  ([#799](https://github.com/n24q02m/qwen3-embed/pull/799),
  [`8367ad7`](https://github.com/n24q02m/qwen3-embed/commit/8367ad71bc3dfe3f51b9a8912b8e86f52478b15a))

- Re-raise worker exceptions after draining the queue + harden worker health detection
  ([`df6c877`](https://github.com/n24q02m/qwen3-embed/commit/df6c877ec390db8339c6857748271dbf83a83391))

- Reduce _rerank_pairs params via kwargs
  ([`2f2c9b1`](https://github.com/n24q02m/qwen3-embed/commit/2f2c9b17db17b45d6264397112122e46d25edbc6))

- Reduce HF download arg count via model param
  ([`cce375f`](https://github.com/n24q02m/qwen3-embed/commit/cce375f5696eaae432a4ecde8761b025b120c236))

- Refresh lockfile (renovate maintenance)
  ([`754b199`](https://github.com/n24q02m/qwen3-embed/commit/754b19934d9411a19426a3467d88ca04e09d9803))

- Remove unused __future__ annotations import in gguf_embedding
  ([`7b96533`](https://github.com/n24q02m/qwen3-embed/commit/7b9653314d909141209b2192260e737686403979))

- Resolve #792 (hermes-solve) ([#793](https://github.com/n24q02m/qwen3-embed/pull/793),
  [`77c95f2`](https://github.com/n24q02m/qwen3-embed/commit/77c95f2670467fc1cf1f83c84cce9fb0eab25cbf))

- Ruff-format test_parallel_processor.py after merge
  ([#873](https://github.com/n24q02m/qwen3-embed/pull/873),
  [`5f6f1b1`](https://github.com/n24q02m/qwen3-embed/commit/5f6f1b198258d874b7f926e4c5d244c4b1febcd6))

- Single-pass real-token count
  ([`cb74697`](https://github.com/n24q02m/qwen3-embed/commit/cb74697744f8c009b587a48cfb98419fb5a42817))

- Split download_files_from_huggingface
  ([`793fc8b`](https://github.com/n24q02m/qwen3-embed/commit/793fc8bf5e960dc76e8e4c85daa3f6c990496ed7))

- Split OnnxTextCrossEncoder __init__ setup
  ([`325341a`](https://github.com/n24q02m/qwen3-embed/commit/325341ac32331586b8c09d8883297cae8f4fe1dd))

- Split retrieve_model_gcs into helpers
  ([`a1c77e0`](https://github.com/n24q02m/qwen3-embed/commit/a1c77e032714b5501f518223a2ca39491e430977))

- Update non-major dependencies
  ([`1bf021b`](https://github.com/n24q02m/qwen3-embed/commit/1bf021bd4b6a7e8d0e508648cae05ce943f0e80f))

- **deps**: Update non-major dependencies ([#782](https://github.com/n24q02m/qwen3-embed/pull/782),
  [`42970bc`](https://github.com/n24q02m/qwen3-embed/commit/42970bcf90a82b9f37451cae702afb1947797ddf))

- **model-management**: Reduce argument count in _download_with_retries and related helpers
  ([`cce375f`](https://github.com/n24q02m/qwen3-embed/commit/cce375f5696eaae432a4ecde8761b025b120c236))

- **utils**: Robust last_token_pool handling for mixed padding and edge cases
  ([`07f19ee`](https://github.com/n24q02m/qwen3-embed/commit/07f19eee2653273aa6517001614c640bc604f6e0))

### Chores

- **deps**: Lock file maintenance ([#801](https://github.com/n24q02m/qwen3-embed/pull/801),
  [`c9daf37`](https://github.com/n24q02m/qwen3-embed/commit/c9daf3798f36eccea5ca23773e48554206cabd47))

- **deps**: Update actions/checkout action to v7
  ([#783](https://github.com/n24q02m/qwen3-embed/pull/783),
  [`3b1691c`](https://github.com/n24q02m/qwen3-embed/commit/3b1691c519593d9b98688513d781e11acede12ba))

- **deps**: Update actions/setup-python digest to ece7cb0
  ([#841](https://github.com/n24q02m/qwen3-embed/pull/841),
  [`f6a8672`](https://github.com/n24q02m/qwen3-embed/commit/f6a867218ed8bfe5474d2bfc13cf1ed386f25377))

- **deps**: Update dawidd6/action-send-mail action to v18
  ([#844](https://github.com/n24q02m/qwen3-embed/pull/844),
  [`703d51c`](https://github.com/n24q02m/qwen3-embed/commit/703d51cae545e7718658da876cdc1c7ac61e7904))


## v1.12.0-beta.3 (2026-06-13)

### Bug Fixes

- Offline cache lookup fails when revision is unpinned
  ([#738](https://github.com/n24q02m/qwen3-embed/pull/738),
  [`e2a3a93`](https://github.com/n24q02m/qwen3-embed/commit/e2a3a93aae11dee25a87a760965db8f801367164))

- **deps**: Update non-major dependencies ([#727](https://github.com/n24q02m/qwen3-embed/pull/727),
  [`39ca34b`](https://github.com/n24q02m/qwen3-embed/commit/39ca34b25b3657a5548f86d31721136d41ebc3e3))

### Chores

- **deps**: Update step-security/harden-runner digest to 9af89fc
  ([#726](https://github.com/n24q02m/qwen3-embed/pull/726),
  [`6c67087`](https://github.com/n24q02m/qwen3-embed/commit/6c67087307a4cfe05b4c39f12fd23b02cc362703))


## v1.12.0-beta.2 (2026-06-12)

### Features

- Add CustomRerankerSpec one-call BYO reranker registration
  ([#736](https://github.com/n24q02m/qwen3-embed/pull/736),
  [`3520b57`](https://github.com/n24q02m/qwen3-embed/commit/3520b576891b9e46f4d885e668d90c3574803278))

- Sync cross-promo section ([#735](https://github.com/n24q02m/qwen3-embed/pull/735),
  [`97cdf07`](https://github.com/n24q02m/qwen3-embed/commit/97cdf0768224cee367e8f687b609fe0f1c0bf2dc))


## v1.12.0-beta.1 (2026-06-12)

### Bug Fixes

- Add NOTICE retaining fastembed (Qdrant) attribution
  ([#733](https://github.com/n24q02m/qwen3-embed/pull/733),
  [`5220b34`](https://github.com/n24q02m/qwen3-embed/commit/5220b3494dc89599bdd7b4f115f251d939cf4a94))

- Format test_export.py ([#733](https://github.com/n24q02m/qwen3-embed/pull/733),
  [`5220b34`](https://github.com/n24q02m/qwen3-embed/commit/5220b3494dc89599bdd7b4f115f251d939cf4a94))

- Honor dim/MRL truncation on custom and pooled embedding paths
  ([#733](https://github.com/n24q02m/qwen3-embed/pull/733),
  [`5220b34`](https://github.com/n24q02m/qwen3-embed/commit/5220b3494dc89599bdd7b4f115f251d939cf4a94))

- Make custom embedding models work under multiprocessing and case-insensitive
  ([#733](https://github.com/n24q02m/qwen3-embed/pull/733),
  [`5220b34`](https://github.com/n24q02m/qwen3-embed/commit/5220b3494dc89599bdd7b4f115f251d939cf4a94))

- Remove orphaned Qodo pr-agent config ([#731](https://github.com/n24q02m/qwen3-embed/pull/731),
  [`bb29b63`](https://github.com/n24q02m/qwen3-embed/commit/bb29b63bd602a641d002b6e756b4122e61138e46))

- Restore PSR changelog generation and backfill version history
  ([#732](https://github.com/n24q02m/qwen3-embed/pull/732),
  [`d450791`](https://github.com/n24q02m/qwen3-embed/commit/d450791c69f070d23948af1287eda4b67ef93739))

- Skip parallel custom-model integration test on Windows spawn deadlock
  ([#733](https://github.com/n24q02m/qwen3-embed/pull/733),
  [`5220b34`](https://github.com/n24q02m/qwen3-embed/commit/5220b3494dc89599bdd7b4f115f251d939cf4a94))

### Features

- Add CustomModelSpec one-call BYO registration helper
  ([#733](https://github.com/n24q02m/qwen3-embed/pull/733),
  [`5220b34`](https://github.com/n24q02m/qwen3-embed/commit/5220b3494dc89599bdd7b4f115f251d939cf4a94))

- Add HF-id to ONNX export helper with lazy optional deps
  ([#733](https://github.com/n24q02m/qwen3-embed/pull/733),
  [`5220b34`](https://github.com/n24q02m/qwen3-embed/commit/5220b3494dc89599bdd7b4f115f251d939cf4a94))

- Document CustomModelSpec bring-your-own-model usage
  ([#733](https://github.com/n24q02m/qwen3-embed/pull/733),
  [`5220b34`](https://github.com/n24q02m/qwen3-embed/commit/5220b3494dc89599bdd7b4f115f251d939cf4a94))

- Production-grade bring-your-own-model (BYO) support
  ([#733](https://github.com/n24q02m/qwen3-embed/pull/733),
  [`5220b34`](https://github.com/n24q02m/qwen3-embed/commit/5220b3494dc89599bdd7b4f115f251d939cf4a94))


## v1.11.2-beta.3 (2026-06-11)

### Bug Fixes

- Add strict= to zip in reranker batch-invariance test (B905)
  ([#730](https://github.com/n24q02m/qwen3-embed/pull/730),
  [`9b0860e`](https://github.com/n24q02m/qwen3-embed/commit/9b0860e2a6affb4aa7fcb4eb3161db185a4c9378))

- Document reranker batch-invariance contract
  ([#730](https://github.com/n24q02m/qwen3-embed/pull/730),
  [`9b0860e`](https://github.com/n24q02m/qwen3-embed/commit/9b0860e2a6affb4aa7fcb4eb3161db185a4c9378))

- Reranker batch-invariant scores (#725) ([#730](https://github.com/n24q02m/qwen3-embed/pull/730),
  [`9b0860e`](https://github.com/n24q02m/qwen3-embed/commit/9b0860e2a6affb4aa7fcb4eb3161db185a4c9378))

- Score reranker texts per-row so scores are batch-invariant
  ([#730](https://github.com/n24q02m/qwen3-embed/pull/730),
  [`9b0860e`](https://github.com/n24q02m/qwen3-embed/commit/9b0860e2a6affb4aa7fcb4eb3161db185a4c9378))


## v1.11.2-beta.2 (2026-06-10)

### Bug Fixes

- Add caching for _list_supported_models in TextCrossEncoder
  ([#715](https://github.com/n24q02m/qwen3-embed/pull/715),
  [`b4abe65`](https://github.com/n24q02m/qwen3-embed/commit/b4abe6548814ffec38b79a9e521b38d6c9457eea))

### Performance Improvements

- Optimize O(N) list iteration in _check_model_exists
  ([#717](https://github.com/n24q02m/qwen3-embed/pull/717),
  [`9d064ab`](https://github.com/n24q02m/qwen3-embed/commit/9d064abf7cc904da377a23a8e062e56ea248466a))

### Testing

- Add coverage for OverflowError in Qwen3CrossEncoderGGUF
  ([#724](https://github.com/n24q02m/qwen3-embed/pull/724),
  [`e23d7bf`](https://github.com/n24q02m/qwen3-embed/commit/e23d7bfa9d7ef1fd7d3f11b9fc5624bf5c51224e))


## v1.11.2-beta.1 (2026-06-10)

### Bug Fixes

- Correct doc drift in agents, security, conduct, contributing
  ([#714](https://github.com/n24q02m/qwen3-embed/pull/714),
  [`9e6038f`](https://github.com/n24q02m/qwen3-embed/commit/9e6038f1d6fe0dff8b238e6f74420a388565c33f))


## v1.11.1 (2026-06-09)


## v1.11.1-beta.1 (2026-06-09)

### Bug Fixes

- Gitignore bot/merge junk artifacts (*.orig/*.rej/*.patch/*.diff/*.cover/*.bak)
  ([#690](https://github.com/n24q02m/qwen3-embed/pull/690),
  [`363004c`](https://github.com/n24q02m/qwen3-embed/commit/363004c22e0d850bb2d954894d6b30f4fbf57fe6))

- Relax tokenizers floor to >=0.22.0 for transformers co-installability
  ([#692](https://github.com/n24q02m/qwen3-embed/pull/692),
  [`4ef7f41`](https://github.com/n24q02m/qwen3-embed/commit/4ef7f41633b577c7f0f060ee44d85dcede6ab5ff))

- **deps**: Update non-major dependencies ([#694](https://github.com/n24q02m/qwen3-embed/pull/694),
  [`2e3b472`](https://github.com/n24q02m/qwen3-embed/commit/2e3b47288ec10bcb45ac8f6adcd2d8bf7abca4d8))

### Chores

- **deps**: Update codecov/codecov-action action to v7
  ([#695](https://github.com/n24q02m/qwen3-embed/pull/695),
  [`ea46046`](https://github.com/n24q02m/qwen3-embed/commit/ea46046f0d63c6dfb5fe3a512ffb537db4dbee26))


## v1.11.0 (2026-06-07)


## v1.11.0-beta.1 (2026-06-07)

### Bug Fixes

- Add _fetch_repo_files tests for model management
  ([#633](https://github.com/n24q02m/qwen3-embed/pull/633),
  [`106d4f3`](https://github.com/n24q02m/qwen3-embed/commit/106d4f3accd5906ea0d23ace0f7b2cb3abe6db9e))

- Bump idna to 3.17 in lockfile maintenance
  ([#644](https://github.com/n24q02m/qwen3-embed/pull/644),
  [`d348689`](https://github.com/n24q02m/qwen3-embed/commit/d3486899f502e25ad6b456041b067efd75617747))

- Correct over-strict tar member validation on Windows (#635)
  ([#645](https://github.com/n24q02m/qwen3-embed/pull/645),
  [`410e4b3`](https://github.com/n24q02m/qwen3-embed/commit/410e4b3a0f85aa55601b017568cc13aab6d56c51))

- Update huggingface-hub to 1.17.0 and ruff to 0.15.15
  ([#643](https://github.com/n24q02m/qwen3-embed/pull/643),
  [`5cb6c9d`](https://github.com/n24q02m/qwen3-embed/commit/5cb6c9d26b0b6d7cb1bdfd95f74e692377b81dd1))

- **deps**: Lock file maintenance
  ([`df1459d`](https://github.com/n24q02m/qwen3-embed/commit/df1459dd2584514e05a22316c7f5d42cbbe79460))

- **deps**: Update actions/checkout digest to df4cb1c
  ([`065f01a`](https://github.com/n24q02m/qwen3-embed/commit/065f01a3c922f568aeedb31dca5909a62d2ee5b3))

- **deps**: Update non-major dependencies
  ([`ea41108`](https://github.com/n24q02m/qwen3-embed/commit/ea411080980537b9165125c57b55e98248857a74))

### Features

- Add test for check_input_length with extremely long input
  ([`2c94b22`](https://github.com/n24q02m/qwen3-embed/commit/2c94b220bce8bf7bb015b57a88895c0301057a45))

- Add test for parallel processor worker exception handling
  ([`3c8c89a`](https://github.com/n24q02m/qwen3-embed/commit/3c8c89a3b377853d48461225e046589e0ad44159))

- Add test for ValueError in model_management._fetch_repo_files
  ([`0673e03`](https://github.com/n24q02m/qwen3-embed/commit/0673e03d0aea771969cae41ba6dc5bacad620744))

- Add tests for ImportError in _check_llama_cpp
  ([`6e5c087`](https://github.com/n24q02m/qwen3-embed/commit/6e5c087df2f59646dd7f424d1ea21397dde396b8))

- Add tests for model snapshot metadata parsing errors
  ([`dd6a120`](https://github.com/n24q02m/qwen3-embed/commit/dd6a120c9fd63311e7a5655a9b0ea88139f13772))

- Add tests for OverflowError in gguf_cross_encoder._score_text
  ([`773f52b`](https://github.com/n24q02m/qwen3-embed/commit/773f52b63ac7213972681d3cf56dbff9c231c506))

### Refactoring

- Reduce cyclomatic complexity in download_model
  ([#639](https://github.com/n24q02m/qwen3-embed/pull/639),
  [`04800e8`](https://github.com/n24q02m/qwen3-embed/commit/04800e8aea54704df0cfbb91801615dd7ab44304))


## v1.10.1 (2026-05-28)

### Bug Fixes

- **deps**: Update non-major dependencies ([#629](https://github.com/n24q02m/qwen3-embed/pull/629),
  [`c32d911`](https://github.com/n24q02m/qwen3-embed/commit/c32d911955f63ede978f2f32ebffb7c0577e1105))

### Chores

- **deps**: Lock file maintenance ([#630](https://github.com/n24q02m/qwen3-embed/pull/630),
  [`b5f41cb`](https://github.com/n24q02m/qwen3-embed/commit/b5f41cbd3c86edfe11a4d3c6ff209b7dfb29f702))


## v1.10.0 (2026-05-26)

### Chores

- **deps**: Lock file maintenance ([#625](https://github.com/n24q02m/qwen3-embed/pull/625),
  [`deb5fe2`](https://github.com/n24q02m/qwen3-embed/commit/deb5fe2abf740021a1e6c902d6b5a3eb60ea67f7))


## v1.10.0-beta.1 (2026-05-24)

### Bug Fixes

- Drop unused providers/device_ids/lazy_load params in GgufEmbedding __init__
  ([`5a57c96`](https://github.com/n24q02m/qwen3-embed/commit/5a57c969c465665c8006a494d6de7a5a5b20d94c))

- Restrict tarfile members to reg/dir/sym/link types + emulate data filter
  ([`7f31c35`](https://github.com/n24q02m/qwen3-embed/commit/7f31c35991f9081631f916d718f7cd9f0228b051))

- Ruff format test_model_description.py for CI green
  ([`c5f2e09`](https://github.com/n24q02m/qwen3-embed/commit/c5f2e09bd12c13ba878cbe11d73af5fdfbc6249e))

- Simplify __init__ parameter list in onnx_text_model + parallel_processor
  ([`c448f77`](https://github.com/n24q02m/qwen3-embed/commit/c448f77fb0c6fcdce53d07ffa34753193211d427))

- Simplify _download_from_gcs parameter list via dataclass
  ([`63898f1`](https://github.com/n24q02m/qwen3-embed/commit/63898f1be479717a167927d4bfe2efeabab9d9a3))

- Simplify _instantiate_onnx_session parameter list
  ([`2b60926`](https://github.com/n24q02m/qwen3-embed/commit/2b60926a29088d7aa81be1759344571d3fe482ac))

- Simplify add_custom_model parameter list via config dataclass
  ([`f109006`](https://github.com/n24q02m/qwen3-embed/commit/f109006a0271500287a193b247b010b4f9ab9562))

- Simplify CustomTextEmbedding __init__ parameter list
  ([`49022e7`](https://github.com/n24q02m/qwen3-embed/commit/49022e78b9b4456d6a1d10c9510baddde8138bef))

- Simplify gguf_cross_encoder + gguf_embedding __init__ parameter list
  ([`786423e`](https://github.com/n24q02m/qwen3-embed/commit/786423ee2b4652ebc79993952fdad9b31404e13d))

- Use dict map for O(1) repo_files lookup in _verify_files_from_metadata
  ([`a2b373d`](https://github.com/n24q02m/qwen3-embed/commit/a2b373d7ec9b2e4d42eb43ad96003d6aa624c510))

- Use math.exp instead of np.exp for single scalar sigmoid in cross_encoder
  ([`f38f489`](https://github.com/n24q02m/qwen3-embed/commit/f38f489729e5c3005a60403ff81a366d2426c253))

- **deps**: Bump huggingface-hub to >=1.14.0,<2.0
  ([`c7c81af`](https://github.com/n24q02m/qwen3-embed/commit/c7c81af291a2cb66c99e4ad5994df23bc991aee5))

- **deps**: Bump urllib3 from 2.6.3 to 2.7.0
  ([#592](https://github.com/n24q02m/qwen3-embed/pull/592),
  [`a7a0837`](https://github.com/n24q02m/qwen3-embed/commit/a7a08370173dbd72034faea638e1094a466b702f))

- **deps**: Refresh uv lock file maintenance
  ([#611](https://github.com/n24q02m/qwen3-embed/pull/611),
  [`9977e93`](https://github.com/n24q02m/qwen3-embed/commit/9977e9335ad8622bda43efc8c0c4616cc121db6f))

- **deps**: Update non-major dependencies ([#590](https://github.com/n24q02m/qwen3-embed/pull/590),
  [`0bcba87`](https://github.com/n24q02m/qwen3-embed/commit/0bcba87c96a415669347978715a73124ce2a4074))

- **security**: Iterate tar members to prevent decompression-bomb OOM
  ([#617](https://github.com/n24q02m/qwen3-embed/pull/617),
  [`3320879`](https://github.com/n24q02m/qwen3-embed/commit/33208794e66ff0a746c5d7401acc167fbd6fa568))

- **security**: Prevent symlink-based permission modification on cache dir
  ([#585](https://github.com/n24q02m/qwen3-embed/pull/585),
  [`f8bcd13`](https://github.com/n24q02m/qwen3-embed/commit/f8bcd13dd98e18c1f18d538f29fce35f2c1cb16d))

- **security**: Thread-safe requests session initialization
  ([#588](https://github.com/n24q02m/qwen3-embed/pull/588),
  [`c094c9b`](https://github.com/n24q02m/qwen3-embed/commit/c094c9bf824f4b9917842a132bd688179e5f0a05))

- **text**: Reduce parameter count and fix formatting
  ([`49022e7`](https://github.com/n24q02m/qwen3-embed/commit/49022e78b9b4456d6a1d10c9510baddde8138bef))

- **text**: Reduce parameter count in CustomTextEmbedding.__init__
  ([`49022e7`](https://github.com/n24q02m/qwen3-embed/commit/49022e78b9b4456d6a1d10c9510baddde8138bef))

### Chores

- **deps**: Lock file maintenance ([#619](https://github.com/n24q02m/qwen3-embed/pull/619),
  [`4806929`](https://github.com/n24q02m/qwen3-embed/commit/4806929b91fc0e00c3971400f3e8bb95efe3a0dd))

- **deps**: Update actions/create-github-app-token digest to bcd2ba4
  ([#597](https://github.com/n24q02m/qwen3-embed/pull/597),
  [`6d2efbd`](https://github.com/n24q02m/qwen3-embed/commit/6d2efbd1be1b45351a3a8630894c57850974ccc5))

- **deps**: Update actions/dependency-review-action action to v5
  ([#586](https://github.com/n24q02m/qwen3-embed/pull/586),
  [`df81194`](https://github.com/n24q02m/qwen3-embed/commit/df811949fb4af6cebe75fed0c37413ef432d3575))

- **deps**: Update codecov/codecov-action digest to e79a696
  ([#618](https://github.com/n24q02m/qwen3-embed/pull/618),
  [`e5db9bf`](https://github.com/n24q02m/qwen3-embed/commit/e5db9bf016bbcf64f2335ff225beb549eefe1e27))

- **deps**: Update dependency ty to >=0.0.39
  ([#622](https://github.com/n24q02m/qwen3-embed/pull/622),
  [`be96922`](https://github.com/n24q02m/qwen3-embed/commit/be969227e475c36c6b3dc6baa637fb03810e99fd))

- **deps**: Update step-security/harden-runner digest to ab7a940
  ([#602](https://github.com/n24q02m/qwen3-embed/pull/602),
  [`dc0f13c`](https://github.com/n24q02m/qwen3-embed/commit/dc0f13c7f1ed37f3e17afab059754d6ea0377dc3))

### Features

- Add missing-value test for deprecated_tar_struct ModelSource property
  ([`8b5caed`](https://github.com/n24q02m/qwen3-embed/commit/8b5caedf96577bad8e2be6c124b17b7c0d6ecfb8))

- Add Table of contents heading + auto-generated link list (Spec E Wave 2)
  ([`42f126d`](https://github.com/n24q02m/qwen3-embed/commit/42f126d8afdf43e55b0acb6f68b6f2f28d0bcf42))

- Add test for ModelManager _get_expected_md5
  ([`8d2b472`](https://github.com/n24q02m/qwen3-embed/commit/8d2b4721c09ab7728972b8bebfb2af7c225e5ec0))

- Add test for PooledNormalizedEmbeddingWorker init_embedding
  ([`a3caf3c`](https://github.com/n24q02m/qwen3-embed/commit/a3caf3cb807890d9dc7ed5060044ff2d014d6fcd))

- Sync cross-promo section ([#583](https://github.com/n24q02m/qwen3-embed/pull/583),
  [`30d0add`](https://github.com/n24q02m/qwen3-embed/commit/30d0add0985dd9512ae32df8378433eb7b4befb8))

### Refactoring

- Reduce parameters in add_custom_model
  ([`f109006`](https://github.com/n24q02m/qwen3-embed/commit/f109006a0271500287a193b247b010b4f9ab9562))

- Reduce parameters in add_custom_model and fix lint
  ([`f109006`](https://github.com/n24q02m/qwen3-embed/commit/f109006a0271500287a193b247b010b4f9ab9562))

### Testing

- **text**: Add unit test for PooledNormalizedEmbeddingWorker.init_embedding
  ([`a3caf3c`](https://github.com/n24q02m/qwen3-embed/commit/a3caf3cb807890d9dc7ed5060044ff2d014d6fcd))


## v1.9.2 (2026-05-06)


## v1.9.2-beta.1 (2026-05-06)

### Bug Fixes

- Align Python version statements with pyproject.toml requires-python
  ([#545](https://github.com/n24q02m/qwen3-embed/pull/545),
  [`e5b54ec`](https://github.com/n24q02m/qwen3-embed/commit/e5b54ec52dc3784b42c005fa872b953c47343e9a))

- In-place division in mean_pooling for ~15-20% perf gain
  ([#548](https://github.com/n24q02m/qwen3-embed/pull/548),
  [`fa280b1`](https://github.com/n24q02m/qwen3-embed/commit/fa280b1eca976940893a65052e300b28f5370f64))

- **deps**: Update dawidd6/action-send-mail action to v17
  ([#547](https://github.com/n24q02m/qwen3-embed/pull/547),
  [`58a700c`](https://github.com/n24q02m/qwen3-embed/commit/58a700c66a6e1be78f8ff6d53ba083edf24ad412))

- **deps**: Update dependency llama-cpp-python to >=0.3.22
  ([#558](https://github.com/n24q02m/qwen3-embed/pull/558),
  [`528f425`](https://github.com/n24q02m/qwen3-embed/commit/528f4250ffb88e3f08590fbe18bc2d48ce1054cc))

- **deps**: Update dependency ty to >=0.0.33
  ([#546](https://github.com/n24q02m/qwen3-embed/pull/546),
  [`ddba364`](https://github.com/n24q02m/qwen3-embed/commit/ddba364c1cf60d8083610fab844d508d40d84eb3))

- **deps**: Update non-major dependencies ([#549](https://github.com/n24q02m/qwen3-embed/pull/549),
  [`8c8c2a5`](https://github.com/n24q02m/qwen3-embed/commit/8c8c2a5a3d1f77ea3f2fdd3cdd614261d8265203))

- **deps**: Update non-major dependencies ([#543](https://github.com/n24q02m/qwen3-embed/pull/543),
  [`e51677b`](https://github.com/n24q02m/qwen3-embed/commit/e51677b09b74c606af068e40eee96a65303b1842))

### Chores

- **deps**: Update step-security/harden-runner digest to a5ad31d
  ([#555](https://github.com/n24q02m/qwen3-embed/pull/555),
  [`82fa9c9`](https://github.com/n24q02m/qwen3-embed/commit/82fa9c9f47d55f654a43323f37a49dddf71c932a))


## v1.9.1 (2026-04-27)

### Bug Fixes

- Scope CD notify-downstream app token to n24q02m profile repo
  ([`957bb1a`](https://github.com/n24q02m/qwen3-embed/commit/957bb1adccbc1d1622937d70082df44a832b7926))

- Sweep doppler/infisical refs to skret SSM
  ([`80eff17`](https://github.com/n24q02m/qwen3-embed/commit/80eff17ec52d8151a125b6c8ddd9e145d549258e))

- **deps**: Update non-major dependencies ([#535](https://github.com/n24q02m/qwen3-embed/pull/535),
  [`0742e96`](https://github.com/n24q02m/qwen3-embed/commit/0742e96970a89b59dead891f49275603290d0640))

### Chores

- **deps**: Update astral-sh/setup-uv action to v8
  ([#532](https://github.com/n24q02m/qwen3-embed/pull/532),
  [`1964aa9`](https://github.com/n24q02m/qwen3-embed/commit/1964aa9556232a06cc7196c111954eac805266b3))


## v1.9.0 (2026-04-21)

### Bug Fixes

- Add diacritic preservation pre-commit hook
  ([#521](https://github.com/n24q02m/qwen3-embed/pull/521),
  [`31bac94`](https://github.com/n24q02m/qwen3-embed/commit/31bac94a10103cafc7b9eda55647dff8ff647afc))

- Add tests for text_cross_encoder_base ([#474](https://github.com/n24q02m/qwen3-embed/pull/474),
  [`4d76cfa`](https://github.com/n24q02m/qwen3-embed/commit/4d76cfab8ea9a95a1719f103ceef50350de051fb))

- Batch ONNX inference in Qwen3CrossEncoder._onnx_embed_texts
  ([#513](https://github.com/n24q02m/qwen3-embed/pull/513),
  [`93704ad`](https://github.com/n24q02m/qwen3-embed/commit/93704ada37618129d00ac06531184a894f5dc17a))

- Bump dev dependency ty to 0.0.32
  ([`fbb6e31`](https://github.com/n24q02m/qwen3-embed/commit/fbb6e31bb294d5f6c203c1c3a6f78e09ce42d01e))

- Bump requires-python to 3.11 for numpy 2.4 compat
  ([`be15ee9`](https://github.com/n24q02m/qwen3-embed/commit/be15ee96d9675cd3381bddf6889f5b50aa5ca913))

- Bump step-security/harden-runner digest to 8d3c67d
  ([`00cecfd`](https://github.com/n24q02m/qwen3-embed/commit/00cecfd5cf72ec7c8e5cda627cd98741c132cc24))

- Drop Python 3.10 from CI matrix after requires-python bump
  ([`5248b3d`](https://github.com/n24q02m/qwen3-embed/commit/5248b3d191109e8c70340ef521f8787cf1fcc42b))

- Enforce iter_batch upper boundary limits ([#468](https://github.com/n24q02m/qwen3-embed/pull/468),
  [`b14bb91`](https://github.com/n24q02m/qwen3-embed/commit/b14bb9182e515a1a7106447727e968e3b57ba69c))

- Ignore coverage.xml and htmlcov artifacts
  ([`117bf8d`](https://github.com/n24q02m/qwen3-embed/commit/117bf8d63419be5a766577a95dc454684c74b2e6))

- Left-pad reranker batches + last-non-pad pooling in yes/no scoring
  ([`b9a5aa6`](https://github.com/n24q02m/qwen3-embed/commit/b9a5aa60458fa73f31b5210b367428ff11a2a632))

- Make path assertions platform-agnostic for Windows CI
  ([`82f688a`](https://github.com/n24q02m/qwen3-embed/commit/82f688a5c760ec2cd95f92d07bad903138598680))

- Optimize iter_batch list slicing by branching list vs tuple
  ([`6079dd9`](https://github.com/n24q02m/qwen3-embed/commit/6079dd9b18951bfaf69746632d4cbd2790a1f1fb))

- Prevent SSRF via open redirects in GCS model downloads
  ([`57f27d5`](https://github.com/n24q02m/qwen3-embed/commit/57f27d5556652401852e956d3bd6bf1e607f185e))

- Reduce code duplication in _load_onnx_model and add parallel execution support
  ([#520](https://github.com/n24q02m/qwen3-embed/pull/520),
  [`f6fd794`](https://github.com/n24q02m/qwen3-embed/commit/f6fd7948ac4a20cb062c808f13798c403a6b3a63))

- Refactor overly long download_files_from_huggingface into helpers
  ([#509](https://github.com/n24q02m/qwen3-embed/pull/509),
  [`1f31b51`](https://github.com/n24q02m/qwen3-embed/commit/1f31b5119b914bac8fc607824da4a15312cd6844))

- Remove AI traces (.jules / superpowers content — belongs in private n24q02m/.superpower repo)
  ([`c167f6a`](https://github.com/n24q02m/qwen3-embed/commit/c167f6a4f4cd6187c285f5c5b6b4bea33b182ede))

- Remove emoji from source code to fix Windows CI
  ([#468](https://github.com/n24q02m/qwen3-embed/pull/468),
  [`b14bb91`](https://github.com/n24q02m/qwen3-embed/commit/b14bb9182e515a1a7106447727e968e3b57ba69c))

- Sync local changes from workspace
  ([`978ca96`](https://github.com/n24q02m/qwen3-embed/commit/978ca96f597296bf74e4b954920890bb4ea6f500))

- Use sqrt(v.dot(v)) for 1D L2 norm in gguf embedding
  ([`3d0f321`](https://github.com/n24q02m/qwen3-embed/commit/3d0f3215581234c55070dc281e224800fde7cf1d))

- **deps**: Bump actions/create-github-app-token digest to 1b10c78
  ([#492](https://github.com/n24q02m/qwen3-embed/pull/492),
  [`57a9aac`](https://github.com/n24q02m/qwen3-embed/commit/57a9aacdc237861393673b28903b7d6250c36f4e))

- **deps**: Bump non-major dependencies (huggingface-hub 1.11, ruff, ty)
  ([#469](https://github.com/n24q02m/qwen3-embed/pull/469),
  [`39f0fb9`](https://github.com/n24q02m/qwen3-embed/commit/39f0fb99b534fb71dda6af181d586b8d6e4255be))

- **deps**: Bump step-security/harden-runner digest to 6c3c2f2
  ([#470](https://github.com/n24q02m/qwen3-embed/pull/470),
  [`f75777a`](https://github.com/n24q02m/qwen3-embed/commit/f75777a071ddba7990517f541041281d4bb17bee))

- **deps**: Update non-major dependencies ([#467](https://github.com/n24q02m/qwen3-embed/pull/467),
  [`8ff84d8`](https://github.com/n24q02m/qwen3-embed/commit/8ff84d826ac2c458e6cc8a0cb65106a8be973892))

### Chores

- Ignore AI assistant traces
  ([`8b7ff69`](https://github.com/n24q02m/qwen3-embed/commit/8b7ff695c31390992ebf9a51ad1e62642c700ab6))

### Features

- Add cross-OS CI matrix (ubuntu/windows/macos)
  ([`493c64a`](https://github.com/n24q02m/qwen3-embed/commit/493c64ac1646a957df29c789541596648a4f68b3))

- Add tests for TextEmbedding.embed delegation
  ([`05ed227`](https://github.com/n24q02m/qwen3-embed/commit/05ed2279224e8287bde7631d3dab001e612144d5))

- Add tests for TextEmbedding.passage_embed and query_embed delegation
  ([`ab46efe`](https://github.com/n24q02m/qwen3-embed/commit/ab46efe35079c825ceee0d7080ac98a27753cfd7))

- Auto-create downstream bump issues on stable release
  ([`5da0d78`](https://github.com/n24q02m/qwen3-embed/commit/5da0d78903d77a8fe8b25c102db0625e8936511b))

- Cache TextEmbedding/TextCrossEncoder model description lookups (O(1))
  ([`145e5d9`](https://github.com/n24q02m/qwen3-embed/commit/145e5d93e52bd515d3e7af6a185a2cd26a40e1a9))

- Migrate code review from Qodo to CodeRabbit
  ([#426](https://github.com/n24q02m/qwen3-embed/pull/426),
  [`b1a4136`](https://github.com/n24q02m/qwen3-embed/commit/b1a4136a15aa7fcefea98463f144762e1949fb1f))

- Replace 2-class softmax with sigmoid on logit difference in cross encoders
  ([`904fe36`](https://github.com/n24q02m/qwen3-embed/commit/904fe36b6f9038f179188e19a103bc96cb7451ad))

- **model_management**: Add desc to tqdm progress bar
  ([#400](https://github.com/n24q02m/qwen3-embed/pull/400),
  [`c0ddd14`](https://github.com/n24q02m/qwen3-embed/commit/c0ddd146b6971c11eaac0d36da99f2890d23b2a0))

### Performance Improvements

- Optimize GCS download chunk size to 1MB ([#516](https://github.com/n24q02m/qwen3-embed/pull/516),
  [`c4b0805`](https://github.com/n24q02m/qwen3-embed/commit/c4b080520ce66b7ecc595c2f4375e240deb99f86))

- Optimize iter_batch for indexable sequences
  ([#468](https://github.com/n24q02m/qwen3-embed/pull/468),
  [`b14bb91`](https://github.com/n24q02m/qwen3-embed/commit/b14bb9182e515a1a7106447727e968e3b57ba69c))

### Testing

- Add coverage for _download_from_hf exception handling
  ([#502](https://github.com/n24q02m/qwen3-embed/pull/502),
  [`aad1270`](https://github.com/n24q02m/qwen3-embed/commit/aad12709a7c5886dc08e8c4648a563174aaea694))

- Add coverage for text_embedding_base.py ([#476](https://github.com/n24q02m/qwen3-embed/pull/476),
  [`aaacbb3`](https://github.com/n24q02m/qwen3-embed/commit/aaacbb328509edc697b41d3a9cdb00fcaa494e6d))

- Fix missing coverage for decompress_to_cache
  ([#514](https://github.com/n24q02m/qwen3-embed/pull/514),
  [`94a8ccf`](https://github.com/n24q02m/qwen3-embed/commit/94a8ccfa253ca6242f2fef8f0ce0a728fc19f3f3))

- **utils**: Add tests for input length validation
  ([#517](https://github.com/n24q02m/qwen3-embed/pull/517),
  [`3f317b9`](https://github.com/n24q02m/qwen3-embed/commit/3f317b944eecc506ed05a54f6b4093c6f1a4475a))


## v1.8.0 (2026-04-04)

### Bug Fixes

- Hardlink path traversal bypass in tarfile extraction
  ([#395](https://github.com/n24q02m/qwen3-embed/pull/395),
  [`c1d0c76`](https://github.com/n24q02m/qwen3-embed/commit/c1d0c76c7d42abd265a0ff977af3419a0105a24b))

### Features

- Notify downstream repos on stable release
  ([#395](https://github.com/n24q02m/qwen3-embed/pull/395),
  [`c1d0c76`](https://github.com/n24q02m/qwen3-embed/commit/c1d0c76c7d42abd265a0ff977af3419a0105a24b))


## v1.7.0 (2026-04-03)

### Bug Fixes

- Consolidated Jules PR review ([#367](https://github.com/n24q02m/qwen3-embed/pull/367),
  [`fa6fa56`](https://github.com/n24q02m/qwen3-embed/commit/fa6fa56649521a6eded2f0fca1c39e3b7dec56dd))

### Continuous Integration

- Sync workflow SHA pins, Qodo vertex_ai config, and VERTEXAI_LOCATION
  ([`5a199b5`](https://github.com/n24q02m/qwen3-embed/commit/5a199b55acf3c962c62822a5d1be6de18da6f73d))

### Features

- Notify downstream repos on stable release
  ([#366](https://github.com/n24q02m/qwen3-embed/pull/366),
  [`cf04fc2`](https://github.com/n24q02m/qwen3-embed/commit/cf04fc2fee555fe865fd6e169042efcbfda0bab9))


## v1.6.0 (2026-03-31)

### Bug Fixes

- Pin pre-commit hooks to commit SHA
  ([`7f64ff1`](https://github.com/n24q02m/qwen3-embed/commit/7f64ff1f5eea5f27f0599f8c2fcd5d2317d7af66))

- Pin third-party GitHub Actions to SHA hashes
  ([`7517dbf`](https://github.com/n24q02m/qwen3-embed/commit/7517dbfe9ed87fe1a838ca73cef41f623ec9c8e7))

- Remove pr-title-check job from CI
  ([`68ec2c9`](https://github.com/n24q02m/qwen3-embed/commit/68ec2c9bbdd3b988a2e8d72b88ad4a220161d0f4))

- Standardize README structure
  ([`412ba62`](https://github.com/n24q02m/qwen3-embed/commit/412ba627f3cc5a11adfce2d79e3172a0fc9c049e))

- **cd**: Remove empty env blocks from OIDC migration
  ([`5c031bd`](https://github.com/n24q02m/qwen3-embed/commit/5c031bdd1e8526a33d0a5ebca3be5f9657595724))

- **cd**: Replace GH_PAT with GitHub App installation token
  ([`bd47ba1`](https://github.com/n24q02m/qwen3-embed/commit/bd47ba11d20a3f9af9cea41434de9cfa9a891847))

- **cd**: Use PyPI OIDC trusted publishing instead of PYPI_TOKEN
  ([`c069253`](https://github.com/n24q02m/qwen3-embed/commit/c06925329dd542fc2f67046fd888fe36f6e8cdcf))

- **ci**: Consolidate SMTP_USERNAME and NOTIFY_EMAIL into one secret
  ([`4943bb3`](https://github.com/n24q02m/qwen3-embed/commit/4943bb3488f8f9946b01d2b815068d8ea32e42ae))

- **ci**: Consolidate SMTP_USERNAME+PASSWORD into SMTP_CREDENTIAL
  ([`dccb016`](https://github.com/n24q02m/qwen3-embed/commit/dccb0160b31b6147329495a3717e0ec2d0d0f765))

- **ci**: Use Vertex AI WIF instead of GEMINI_API_KEY for code review
  ([`cbf982d`](https://github.com/n24q02m/qwen3-embed/commit/cbf982df66638f11ae4441bb4840bf75ca89aa3a))

### Chores

- Add .code-review-graph/ to .gitignore
  ([`255afd5`](https://github.com/n24q02m/qwen3-embed/commit/255afd51e20a9a5cd89dd822089cd4e6dc3f9ae6))

- Add .env to .gitignore for secret protection
  ([`1b79f3f`](https://github.com/n24q02m/qwen3-embed/commit/1b79f3fb89915aad0d07fe790779a01c810fd68d))

- Add Infisical project configuration
  ([`a9ec91c`](https://github.com/n24q02m/qwen3-embed/commit/a9ec91c7d65051e6fb14ea6dd5ab79d0e0428793))

- Remove Infisical config (empty project deleted)
  ([`f540f68`](https://github.com/n24q02m/qwen3-embed/commit/f540f68f7d5ff868b4e60ef1556ac1e29303cfc8))

- **deps**: Lock file maintenance
  ([`f03b866`](https://github.com/n24q02m/qwen3-embed/commit/f03b866422af0ed9743649a8ce427fa6a946c80e))

- **deps**: Pin dependencies ([#302](https://github.com/n24q02m/qwen3-embed/pull/302),
  [`86f1afc`](https://github.com/n24q02m/qwen3-embed/commit/86f1afcbc8087b04eabcb7a47636487fb2597bf2))

- **deps**: Update actions/create-github-app-token action to v3
  ([#304](https://github.com/n24q02m/qwen3-embed/pull/304),
  [`c3e23e2`](https://github.com/n24q02m/qwen3-embed/commit/c3e23e2061d5b7198744eca9909b5fec4471ff2b))

- **deps**: Update codecov/codecov-action action to v6
  ([#305](https://github.com/n24q02m/qwen3-embed/pull/305),
  [`c33a1eb`](https://github.com/n24q02m/qwen3-embed/commit/c33a1eb30b2d333efd35c340d2087ad8a5487922))

- **deps**: Update codecov/codecov-action digest to 75cd116
  ([#298](https://github.com/n24q02m/qwen3-embed/pull/298),
  [`b3c4f40`](https://github.com/n24q02m/qwen3-embed/commit/b3c4f4033a1b83b126c269b4ce55e0c24b13bfaf))

- **deps**: Update dependency requests to v2.33.0 [security]
  ([`e1cead3`](https://github.com/n24q02m/qwen3-embed/commit/e1cead39f7a1b505efd0fb8104f80de9364d370a))

### Documentation

- Add CLAUDE.md with project conventions and structure
  ([`a9f19c8`](https://github.com/n24q02m/qwen3-embed/commit/a9f19c858e69a1c85bbe345ad1536015d9d03f89))

- Fix CLAUDE.md discrepancies
  ([`c963935`](https://github.com/n24q02m/qwen3-embed/commit/c963935ed0cf55213b86da0049e73721612de88d))

### Features

- Notify downstream repos on stable release
  ([`259afd8`](https://github.com/n24q02m/qwen3-embed/commit/259afd881b700b05f479868cfd933a99385ba983))

- Use batched tokenization in Qwen3CrossEncoder
  ([#287](https://github.com/n24q02m/qwen3-embed/pull/287),
  [`9a236f1`](https://github.com/n24q02m/qwen3-embed/commit/9a236f1e8ff997335c39f154ee4eeebb8935d61b))

### Refactoring

- 🧹 Code Health: Refactor export_model to fix overly long function
  ([#315](https://github.com/n24q02m/qwen3-embed/pull/315),
  [`0803463`](https://github.com/n24q02m/qwen3-embed/commit/08034637995bf3504687163caa73f4dc47f5b44d))

### Testing

- Add cache checking fallback tests for model management
  ([#321](https://github.com/n24q02m/qwen3-embed/pull/321),
  [`02288e5`](https://github.com/n24q02m/qwen3-embed/commit/02288e5e1a19fa92aa459c4c1a20451c86d97a09))


## v1.5.1 (2026-03-20)

### Bug Fixes

- Add Device to public exports, update SECURITY.md
  ([`ed5b3e5`](https://github.com/n24q02m/qwen3-embed/commit/ed5b3e5a583c81750ded27f4af19d38aa6964d20))

### Chores

- Align CI/CD action versions
  ([`3cb68f5`](https://github.com/n24q02m/qwen3-embed/commit/3cb68f5f99b778ad64b44ea4f4e72930cbbbfc0d))

- **deps**: Lock file maintenance ([#279](https://github.com/n24q02m/qwen3-embed/pull/279),
  [`29bc4c4`](https://github.com/n24q02m/qwen3-embed/commit/29bc4c4941168498ad344a4a5677c63c66c3121e))

- **deps**: Update codecov/codecov-action digest to 1af5884
  ([#280](https://github.com/n24q02m/qwen3-embed/pull/280),
  [`047c7f4`](https://github.com/n24q02m/qwen3-embed/commit/047c7f41a626741a88f41a6033163a6935928d50))

- **deps**: Update dawidd6/action-send-mail action to v16
  ([#278](https://github.com/n24q02m/qwen3-embed/pull/278),
  [`061b417`](https://github.com/n24q02m/qwen3-embed/commit/061b417272d6aa56dc62443002a3f83f81aa2a0c))


## v1.5.0 (2026-03-18)

### Bug Fixes

- Add debug logging to empty except block in model cache fallback
  ([`9ddda4a`](https://github.com/n24q02m/qwen3-embed/commit/9ddda4ad20bfe6a6be75836d300f68870b099cac))

- Add noqa to unused arguments in gguf_cross_encoder.py
  ([#254](https://github.com/n24q02m/qwen3-embed/pull/254),
  [`64ca958`](https://github.com/n24q02m/qwen3-embed/commit/64ca95811f937a2654ae89c5667f5586e6ebc82c))

- Fix TOKEN_NO_ID mismatch and add input sanitization in GGUF reranker
  ([`9df0c3e`](https://github.com/n24q02m/qwen3-embed/commit/9df0c3e19746af3d745a7c0e959302c547c92972))

- Resolve formatting errors ([#258](https://github.com/n24q02m/qwen3-embed/pull/258),
  [`8abb836`](https://github.com/n24q02m/qwen3-embed/commit/8abb83674b3340869c89cf79cb5bbe94f1089f05))

- Resolve lint errors for unused imports and sorting
  ([#258](https://github.com/n24q02m/qwen3-embed/pull/258),
  [`8abb836`](https://github.com/n24q02m/qwen3-embed/commit/8abb83674b3340869c89cf79cb5bbe94f1089f05))

- Testing] Cover ImportError from builtins.__import__ in _check_llama_cpp
  ([#240](https://github.com/n24q02m/qwen3-embed/pull/240),
  [`465e700`](https://github.com/n24q02m/qwen3-embed/commit/465e7001e3cc014ff7e716b29695c6422a4462ea))

- Testing] Test OSError/ValueError in HF download
  ([#265](https://github.com/n24q02m/qwen3-embed/pull/265),
  [`ee18b90`](https://github.com/n24q02m/qwen3-embed/commit/ee18b90c3030d6432caa38a3277958bc3a190920))

- 🧹 [code health improvement] Fix unused arguments in GGUF cross encoder
  ([#260](https://github.com/n24q02m/qwen3-embed/pull/260),
  [`b99bb6e`](https://github.com/n24q02m/qwen3-embed/commit/b99bb6ea8f99c195634c27d756ff9bf6745da517))

- 🧹 [code health] Format missing files and refactor _load_onnx_model
  ([#264](https://github.com/n24q02m/qwen3-embed/pull/264),
  [`a3e0e99`](https://github.com/n24q02m/qwen3-embed/commit/a3e0e99edca51a6b0aabe5dcc4fc24d43fc61e40))

- 🧹 [code health] Format missing files and resolve typing issues
  ([#264](https://github.com/n24q02m/qwen3-embed/pull/264),
  [`a3e0e99`](https://github.com/n24q02m/qwen3-embed/commit/a3e0e99edca51a6b0aabe5dcc4fc24d43fc61e40))

- 🧹 [code health] Refactor _load_onnx_model to reduce complexity
  ([#264](https://github.com/n24q02m/qwen3-embed/pull/264),
  [`a3e0e99`](https://github.com/n24q02m/qwen3-embed/commit/a3e0e99edca51a6b0aabe5dcc4fc24d43fc61e40))

- 🧹 Code Health: Refactor download_model ([#261](https://github.com/n24q02m/qwen3-embed/pull/261),
  [`e5d9055`](https://github.com/n24q02m/qwen3-embed/commit/e5d9055ff18921356da3f33006d0b0c3bf48aaba))

- 🧹 Suppress unused arg warnings in GGUF cross encoder
  ([#260](https://github.com/n24q02m/qwen3-embed/pull/260),
  [`b99bb6e`](https://github.com/n24q02m/qwen3-embed/commit/b99bb6ea8f99c195634c27d756ff9bf6745da517))

- 🧹 Suppress unused arg warnings in GGUF cross encoder and format code
  ([#260](https://github.com/n24q02m/qwen3-embed/pull/260),
  [`b99bb6e`](https://github.com/n24q02m/qwen3-embed/commit/b99bb6ea8f99c195634c27d756ff9bf6745da517))

- 🧹 Suppress unused argument warnings in Qwen3CrossEncoderGGUF
  ([#254](https://github.com/n24q02m/qwen3-embed/pull/254),
  [`64ca958`](https://github.com/n24q02m/qwen3-embed/commit/64ca95811f937a2654ae89c5667f5586e6ebc82c))

- **ci**: Use pull_request_target for jobs requiring secrets
  ([`015199e`](https://github.com/n24q02m/qwen3-embed/commit/015199e56fc428603ea5218d342805156b6b36b3))

### Chores

- **deps**: Lock file maintenance ([#228](https://github.com/n24q02m/qwen3-embed/pull/228),
  [`d44d363`](https://github.com/n24q02m/qwen3-embed/commit/d44d3631896b8a395f46dff3279f3c1d6047d328))

- **deps**: Update astral-sh/setup-uv digest to e06108d
  ([#225](https://github.com/n24q02m/qwen3-embed/pull/225),
  [`3b44efe`](https://github.com/n24q02m/qwen3-embed/commit/3b44efe0e285dc3b6aaebaa45abbefb074fbf312))

- **deps**: Update dawidd6/action-send-mail action to v14
  ([#233](https://github.com/n24q02m/qwen3-embed/pull/233),
  [`e7f9e91`](https://github.com/n24q02m/qwen3-embed/commit/e7f9e91a58c2fc061606c87189ca328e6995de9a))

### Code Style

- Format code ([#254](https://github.com/n24q02m/qwen3-embed/pull/254),
  [`64ca958`](https://github.com/n24q02m/qwen3-embed/commit/64ca95811f937a2654ae89c5667f5586e6ebc82c))

### Features

- Add missing edge cases for remove_non_alphanumeric
  ([#242](https://github.com/n24q02m/qwen3-embed/pull/242),
  [`1ccae00`](https://github.com/n24q02m/qwen3-embed/commit/1ccae0014325ba17254098ebd110b58e1910bb68))

- Add tests for common types ([#243](https://github.com/n24q02m/qwen3-embed/pull/243),
  [`5d1e462`](https://github.com/n24q02m/qwen3-embed/commit/5d1e4623c3c2e5d068aa9292f4a1c8319993071d))

- Add tests for OnnxTextEmbedding ([#267](https://github.com/n24q02m/qwen3-embed/pull/267),
  [`189d9bc`](https://github.com/n24q02m/qwen3-embed/commit/189d9bc085168ddc9a74ee9810e332665e436334))

- Testing improvement: Add parallel processor exception handling coverage
  ([#258](https://github.com/n24q02m/qwen3-embed/pull/258),
  [`8abb836`](https://github.com/n24q02m/qwen3-embed/commit/8abb83674b3340869c89cf79cb5bbe94f1089f05))

- Testing] Test PooledEmbedding.mean_pooling
  ([#252](https://github.com/n24q02m/qwen3-embed/pull/252),
  [`7db4d0a`](https://github.com/n24q02m/qwen3-embed/commit/7db4d0aff3fb2bf1c5b4d8d02467782c8db53f3c))

### Refactoring

- Extract nested functions from download_files_from_huggingface
  ([#266](https://github.com/n24q02m/qwen3-embed/pull/266),
  [`9df5da8`](https://github.com/n24q02m/qwen3-embed/commit/9df5da8a7e86a67c3aa588f85b52ac29465d2d6a))

### Testing

- Add missing tests for onnx embedding ([#267](https://github.com/n24q02m/qwen3-embed/pull/267),
  [`189d9bc`](https://github.com/n24q02m/qwen3-embed/commit/189d9bc085168ddc9a74ee9810e332665e436334))

- Add model inference tests for Qwen3CrossEncoder
  ([#246](https://github.com/n24q02m/qwen3-embed/pull/246),
  [`d9bd783`](https://github.com/n24q02m/qwen3-embed/commit/d9bd7834dc46666e9e23f7f0d2d6fd74dec47476))

- Add multiprocessing exception handling tests
  ([#258](https://github.com/n24q02m/qwen3-embed/pull/258),
  [`8abb836`](https://github.com/n24q02m/qwen3-embed/commit/8abb83674b3340869c89cf79cb5bbe94f1089f05))

- Add tests for common type aliases and Device enum
  ([#243](https://github.com/n24q02m/qwen3-embed/pull/243),
  [`5d1e462`](https://github.com/n24q02m/qwen3-embed/commit/5d1e4623c3c2e5d068aa9292f4a1c8319993071d))

- Add tests for PooledEmbedding.mean_pooling method
  ([#252](https://github.com/n24q02m/qwen3-embed/pull/252),
  [`7db4d0a`](https://github.com/n24q02m/qwen3-embed/commit/7db4d0aff3fb2bf1c5b4d8d02467782c8db53f3c))

- Fix OnnxTextEmbeddingWorker test to appease type checker
  ([#267](https://github.com/n24q02m/qwen3-embed/pull/267),
  [`189d9bc`](https://github.com/n24q02m/qwen3-embed/commit/189d9bc085168ddc9a74ee9810e332665e436334))

- Fix ruff formatting errors ([#243](https://github.com/n24q02m/qwen3-embed/pull/243),
  [`5d1e462`](https://github.com/n24q02m/qwen3-embed/commit/5d1e4623c3c2e5d068aa9292f4a1c8319993071d))


## v1.4.3 (2026-03-13)

### Bug Fixes

- Disable ORT memory pattern to prevent RAM growth with varying sequences
  ([`c265805`](https://github.com/n24q02m/qwen3-embed/commit/c26580551a9c31c72735f177aee9e2313bd69bc5))

### Documentation

- Add YesNo reranker variant to README
  ([`32ba3f5`](https://github.com/n24q02m/qwen3-embed/commit/32ba3f53c42fec0554d9512eac03d187de51563f))


## v1.4.2 (2026-03-13)

### Bug Fixes

- YesNo model uses same HF repo as other variants
  ([`d6ba630`](https://github.com/n24q02m/qwen3-embed/commit/d6ba630bc678c39da55e228a119ea8f39fb12ee7))


## v1.4.1 (2026-03-13)

### Bug Fixes

- Correct TOKEN_NO_ID (2132→2152) and add optimized YesNo model support
  ([`113cdd7`](https://github.com/n24q02m/qwen3-embed/commit/113cdd739e4b6e070fcc9f4a72eaf5a63e50be01))

### Breaking Changes

- TOKEN_NO_ID correction changes reranker scoring behavior.


## v1.4.0 (2026-03-12)

### Bug Fixes

- Cap onnxruntime <1.24 for Python 3.10 (no wheels available)
  ([`971108e`](https://github.com/n24q02m/qwen3-embed/commit/971108e2de85253b4c7a298e9fdb1645fd3dc202))


## v1.4.0-beta.1 (2026-03-12)

### Bug Fixes

- Disable mise runtime updates in Renovate
  ([`a8b2271`](https://github.com/n24q02m/qwen3-embed/commit/a8b2271847dd0d0a51fb5ccc0ae97052e271e210))

### Documentation

- Add v1.3.0 entry to CHANGELOG
  ([`d7fe8a1`](https://github.com/n24q02m/qwen3-embed/commit/d7fe8a1e47c25c819da44fd394afd6265eeee52d))

### Features

- Support Python 3.10-3.14 (5 latest versions)
  ([`aab129b`](https://github.com/n24q02m/qwen3-embed/commit/aab129b9944823dd1e927296bd9e28862d58fdae))


## v1.3.0 (2026-03-11)

### Bug Fixes

- Add .jules/ and JULES.md to gitignore
  ([`42a9ed1`](https://github.com/n24q02m/qwen3-embed/commit/42a9ed1ddbdc8e47ad66fb115b0870765d3ad8d9))

- Add file integrity verification for GCS downloads
  ([#221](https://github.com/n24q02m/qwen3-embed/pull/221),
  [`3582fbc`](https://github.com/n24q02m/qwen3-embed/commit/3582fbc28c784e03e79a9ed405e0ce8ae0f88334))

- Add MD5 hash verification for GCS file downloads
  ([#221](https://github.com/n24q02m/qwen3-embed/pull/221),
  [`3582fbc`](https://github.com/n24q02m/qwen3-embed/commit/3582fbc28c784e03e79a9ed405e0ce8ae0f88334))

- Correct Qodo PR Agent ignore_pr_authors config
  ([`0ee787e`](https://github.com/n24q02m/qwen3-embed/commit/0ee787e7e12e6be3ffe4c3d92509278f2a209989))

- Fix arbitrary file write via archive extraction (Tar Slip)
  ([#218](https://github.com/n24q02m/qwen3-embed/pull/218),
  [`ce7a251`](https://github.com/n24q02m/qwen3-embed/commit/ce7a25185297e37e4c6ca15d4a0ba3e7a412b927))

- Fix insecure temporary directory creation
  ([#212](https://github.com/n24q02m/qwen3-embed/pull/212),
  [`a2ec057`](https://github.com/n24q02m/qwen3-embed/commit/a2ec0578b4bbafbe08350c3bf44000246ae13408))

- Fix unsafe file download via unvalidated URL (SSRF)
  ([#203](https://github.com/n24q02m/qwen3-embed/pull/203),
  [`35df610`](https://github.com/n24q02m/qwen3-embed/commit/35df610227a1606721b5fd7e181d1c246d63e5a8))

- Improve decompress_to_cache TarError handling test
  ([#193](https://github.com/n24q02m/qwen3-embed/pull/193),
  [`00a48ba`](https://github.com/n24q02m/qwen3-embed/commit/00a48ba7b233013fc7ea11a0f8002fe7119ae875))

- Improve metadata save error logging and add test
  ([#224](https://github.com/n24q02m/qwen3-embed/pull/224),
  [`d84cd31`](https://github.com/n24q02m/qwen3-embed/commit/d84cd31afde9bae806837926b69f5b55da88d988))

- Remove commit-message-check job
  ([`410477d`](https://github.com/n24q02m/qwen3-embed/commit/410477d00548addd314f91d9adc71a4db7997236))

- Revert Python to 3.13, fix Renovate config, add Q4F16/GGUF integration tests, fix model cache
  validation
  ([`3770fe8`](https://github.com/n24q02m/qwen3-embed/commit/3770fe8b1332bb2e769f191bf881eadacd4c349e))

- Standardize CI with PR title check, email notify, and templates
  ([`a4582b5`](https://github.com/n24q02m/qwen3-embed/commit/a4582b57e086b3bdcdd6ffb49926fdd208d98bfe))

- Suppress Bandit B615 on offline snapshot_download
  ([#222](https://github.com/n24q02m/qwen3-embed/pull/222),
  [`306f3b8`](https://github.com/n24q02m/qwen3-embed/commit/306f3b85c4661d0277f3965c2fc3f9796376f891))

- Sync CI/CD configs and standardize templates
  ([`1643dd3`](https://github.com/n24q02m/qwen3-embed/commit/1643dd3e11f9cccfac2b3f5b65d9b4d5bd65fed6))

- Update Codecov badge in README.md
  ([`703a2dd`](https://github.com/n24q02m/qwen3-embed/commit/703a2dde9a7f3005b1d82c90342d8d2a2dda2ec9))

- **ci**: Fix Qodo PR review for external contributors
  ([`4034fab`](https://github.com/n24q02m/qwen3-embed/commit/4034fab11d223d8596c31d91e148c414892b6351))

- **ci**: Pin PSR v10, Python 3.13, Node 24, Java 21 in Renovate
  ([`d07b6be`](https://github.com/n24q02m/qwen3-embed/commit/d07b6bee71383d4863c626a8fe55bc4d7010d580))

- **ci**: Revert PSR v9 downgrade back to v10
  ([`ee8e452`](https://github.com/n24q02m/qwen3-embed/commit/ee8e4527b2142232e6766d64bbab1d1469221634))

- **deps**: Update non-major dependencies ([#166](https://github.com/n24q02m/qwen3-embed/pull/166),
  [`62f2064`](https://github.com/n24q02m/qwen3-embed/commit/62f2064aec7a92fe3ec34a5020b19b8e39aad715))

- **security**: Add SSRF protection, revision pinning, and code quality improvements
  ([`67600c3`](https://github.com/n24q02m/qwen3-embed/commit/67600c31962a4c71948d151ed11be8b548de6e38))

### Chores

- Fix formatting in onnx_model.py
  ([`bf9e7e1`](https://github.com/n24q02m/qwen3-embed/commit/bf9e7e1c5ab581206c6d61d0dcfc057709cfaec3))

- Remove leftover Jules bot files
  ([`3ebcb2e`](https://github.com/n24q02m/qwen3-embed/commit/3ebcb2eace679f209633c87922e5f4150c00528f))

- **deps**: Lock file maintenance ([#121](https://github.com/n24q02m/qwen3-embed/pull/121),
  [`abe9f30`](https://github.com/n24q02m/qwen3-embed/commit/abe9f304945a913910d9a598041830bdc28ce19c))

- **deps**: Pin dawidd6/action-send-mail action
  ([#175](https://github.com/n24q02m/qwen3-embed/pull/175),
  [`0b26da8`](https://github.com/n24q02m/qwen3-embed/commit/0b26da8e3aa7732465b2610040bbbeb6ae86f779))

- **deps**: Pin dependencies ([#119](https://github.com/n24q02m/qwen3-embed/pull/119),
  [`444eab9`](https://github.com/n24q02m/qwen3-embed/commit/444eab94030445cae65dda592f7a1c8269135926))

- **deps**: Update actions/dependency-review-action
  ([#180](https://github.com/n24q02m/qwen3-embed/pull/180),
  [`8830d19`](https://github.com/n24q02m/qwen3-embed/commit/8830d19e6a7e1ddaa06b888987eaf19804a4e103))

### Code Style

- Format test file with ruff ([#221](https://github.com/n24q02m/qwen3-embed/pull/221),
  [`3582fbc`](https://github.com/n24q02m/qwen3-embed/commit/3582fbc28c784e03e79a9ed405e0ce8ae0f88334))

- Format tests with ruff ([#187](https://github.com/n24q02m/qwen3-embed/pull/187),
  [`d3e998c`](https://github.com/n24q02m/qwen3-embed/commit/d3e998c5190ced6a3aaeed64d02e3acc969c9ab3))

- 🧹 Run ruff format on qwen3_embed/parallel_processor.py
  ([`90f6386`](https://github.com/n24q02m/qwen3-embed/commit/90f63869b0eb4cc493625bfdd0cf22cabae8b289))

### Continuous Integration

- Improve PR checks and Qodo filtering ([#179](https://github.com/n24q02m/qwen3-embed/pull/179),
  [`da08e39`](https://github.com/n24q02m/qwen3-embed/commit/da08e392d58a14a25d348fd8ce6187e30ba44f5b))

- Trigger CI run
  ([`57b0209`](https://github.com/n24q02m/qwen3-embed/commit/57b02095c6c02a7b28699d0560bf4f37019182c9))

### Documentation

- Add related projects cross-references
  ([`c2d08ea`](https://github.com/n24q02m/qwen3-embed/commit/c2d08ea4d157e511bdf512714133c6e278f5d2c4))

- Update docs for stable release - Production/Stable status, complete CHANGELOG, accurate README and
  AGENTS.md
  ([`dae3f2f`](https://github.com/n24q02m/qwen3-embed/commit/dae3f2fe3b7a03ba66515e164d673a37a4ed372f))

### Features

- Add coverage for add_extra_session_options
  ([#187](https://github.com/n24q02m/qwen3-embed/pull/187),
  [`d3e998c`](https://github.com/n24q02m/qwen3-embed/commit/d3e998c5190ced6a3aaeed64d02e3acc969c9ab3))

- Add test for _collect_file_metadata function
  ([#186](https://github.com/n24q02m/qwen3-embed/pull/186),
  [`3731db9`](https://github.com/n24q02m/qwen3-embed/commit/3731db9c8b7a85be6296cafa90ede5e084f638a8))

- Add tests for DenseModelDescription valid and invalid dimension limits
  ([#188](https://github.com/n24q02m/qwen3-embed/pull/188),
  [`b204339`](https://github.com/n24q02m/qwen3-embed/commit/b204339c87bb6e85b41927947c8da715e443fca9))

- Add tests for get_all_punctuation utility
  ([#185](https://github.com/n24q02m/qwen3-embed/pull/185),
  [`d8422d1`](https://github.com/n24q02m/qwen3-embed/commit/d8422d15a237ed44f8497015c75aa8bb01a535f2))

- Add unit tests for remove_non_alphanumeric utility
  ([#223](https://github.com/n24q02m/qwen3-embed/pull/223),
  [`c991044`](https://github.com/n24q02m/qwen3-embed/commit/c99104454d66bfbabebd65001f1be312dfc7274e))

### Testing

- Add coverage for add_extra_session_options
  ([#187](https://github.com/n24q02m/qwen3-embed/pull/187),
  [`d3e998c`](https://github.com/n24q02m/qwen3-embed/commit/d3e998c5190ced6a3aaeed64d02e3acc969c9ab3))

- Add MD5 hash verification tests ([#221](https://github.com/n24q02m/qwen3-embed/pull/221),
  [`3582fbc`](https://github.com/n24q02m/qwen3-embed/commit/3582fbc28c784e03e79a9ed405e0ce8ae0f88334))

- Add tests for DenseModelDescription ([#188](https://github.com/n24q02m/qwen3-embed/pull/188),
  [`b204339`](https://github.com/n24q02m/qwen3-embed/commit/b204339c87bb6e85b41927947c8da715e443fca9))

- Add tests for get_all_punctuation utility
  ([#185](https://github.com/n24q02m/qwen3-embed/pull/185),
  [`d8422d1`](https://github.com/n24q02m/qwen3-embed/commit/d8422d15a237ed44f8497015c75aa8bb01a535f2))

- Format code with ruff ([#188](https://github.com/n24q02m/qwen3-embed/pull/188),
  [`b204339`](https://github.com/n24q02m/qwen3-embed/commit/b204339c87bb6e85b41927947c8da715e443fca9))

- Format tests with ruff format ([#185](https://github.com/n24q02m/qwen3-embed/pull/185),
  [`d8422d1`](https://github.com/n24q02m/qwen3-embed/commit/d8422d15a237ed44f8497015c75aa8bb01a535f2))


## v1.2.0 (2026-03-01)

### Bug Fixes

- Add CI status badge to README
  ([`2f1a77a`](https://github.com/n24q02m/qwen3-embed/commit/2f1a77a5a2815ec6c87f31c7cc9d2df03ef3142f))

- Add community files and replace SECURITY.md template
  ([`0efa84a`](https://github.com/n24q02m/qwen3-embed/commit/0efa84a7387b68c8b7796b2f3da08de79dcda1b1))

- Delete .jules directory
  ([`c72bd6d`](https://github.com/n24q02m/qwen3-embed/commit/c72bd6d909d33def2a67409a372b225217fc6ca5))

- Increase test coverage to 92% and resolve typing errors
  ([`7658adf`](https://github.com/n24q02m/qwen3-embed/commit/7658adfad316a6d81ae22f76710cce714d5909f5))

- Make TextCrossEncoder.add_custom_model case-insensitive
  ([#88](https://github.com/n24q02m/qwen3-embed/pull/88),
  [`30212f0`](https://github.com/n24q02m/qwen3-embed/commit/30212f0349dbf3e71919127516e59f72d57a57eb))

- Remove test artifact and ignore *.tar.gz files
  ([`982a6f0`](https://github.com/n24q02m/qwen3-embed/commit/982a6f0eca73a2e15e7a40ec08ff629cccecb8f7))

- Standardize repo structure with enforce-commit hook and ty checker
  ([`b3190c2`](https://github.com/n24q02m/qwen3-embed/commit/b3190c2d1ea83b90ab28611222dc66cff590ef7c))

- Update README badges with Codecov, tech stack, and engineering standards
  ([`b0d02eb`](https://github.com/n24q02m/qwen3-embed/commit/b0d02ebc75a665fb2ddafa563959628d1923ea06))

- Update ruff-pre-commit rev to v0.15.1 and fix formatting
  ([`419dfca`](https://github.com/n24q02m/qwen3-embed/commit/419dfca3fbc1f82c2c5edc96a383617f21924639))

- **ci**: Fix Qodo Merge env variable dot notation bug
  ([`915a0a0`](https://github.com/n24q02m/qwen3-embed/commit/915a0a04ed3fa57e7e123a935be99d744368dc1d))

- **ci**: Fix Qodo model to gemini-3-flash-preview
  ([`e92717b`](https://github.com/n24q02m/qwen3-embed/commit/e92717b64c2dec512cb7f7d37b907193ea5f922f))

- **ci**: Fix syntax errors and correctly configure Qodo + Gemini 3 Flash
  ([`1e89e81`](https://github.com/n24q02m/qwen3-embed/commit/1e89e8188bb10b8b654649919023065cf3f7dfc7))

- **ci**: Move pr-agent config to .pr_agent.toml
  ([`7e11851`](https://github.com/n24q02m/qwen3-embed/commit/7e11851e0d2a9a82358d6670433b47e5f6a4c900))

- **ci**: Update to supported Gemini 3 and 2.5 flash models
  ([`5eaeae7`](https://github.com/n24q02m/qwen3-embed/commit/5eaeae7aaf28ff985046c297bdb10528f54c2f96))

### Chores

- Add Gemini Code Assist style guide
  ([`66c60cc`](https://github.com/n24q02m/qwen3-embed/commit/66c60cc14d78fc0ce1f5438fd053dceb6b2f36b2))

- Apply bulk fixes for performance, security, and type checking
  ([`e90cae9`](https://github.com/n24q02m/qwen3-embed/commit/e90cae95540ae3c8eb1e24edcbe8bc37fb8af0de))

- Change Renovate schedule to daily 5am
  ([`7c0d535`](https://github.com/n24q02m/qwen3-embed/commit/7c0d53531807ed93de1dbe0a576e24c544d68710))

- Remove CodeRabbit config, migrating to Gemini Code Assist
  ([`665d21b`](https://github.com/n24q02m/qwen3-embed/commit/665d21b35eb1fcc6bca02d9527cd8fe9d44a5878))

- Update uv.lock
  ([`b144d88`](https://github.com/n24q02m/qwen3-embed/commit/b144d8874a341615370154df64913abe4d186035))

- **config**: Migrate config renovate.json
  ([`e145c0e`](https://github.com/n24q02m/qwen3-embed/commit/e145c0e4721ba1e0a60cab030a6f2fb1a1dc7ebc))

### Documentation

- Add AGENTS.md for AI coding agents
  ([`68b557c`](https://github.com/n24q02m/qwen3-embed/commit/68b557c1e7ed618d56d3aa0c92d79bc8cbd96f9d))

- Standardize README with ProperCase title and bold tagline
  ([`bdb7e5d`](https://github.com/n24q02m/qwen3-embed/commit/bdb7e5dbc9ee18a2c75c618516f8e77b1eca99f2))

### Features

- Add Codecov coverage upload and CodeRabbit config
  ([`891201b`](https://github.com/n24q02m/qwen3-embed/commit/891201b76ac613c1a6695350b2b1566fb7fe2441))

- Migrate to 2025-2026 tech stack (bun/uv/ty)
  ([`7b689e4`](https://github.com/n24q02m/qwen3-embed/commit/7b689e46fba19b92c80849cda97985e3c5afe1d3))

- **ci**: Add Renovate config for automated dependency updates
  ([`4e307b4`](https://github.com/n24q02m/qwen3-embed/commit/4e307b4f766f1092d73fafcf8722ecea4d496c26))

- **ci**: Add StepSecurity Harden-Runner to all workflow jobs (audit mode)
  ([`78358d5`](https://github.com/n24q02m/qwen3-embed/commit/78358d5a2c200a5c7db43614eec8da4de8c56d29))

- **ci**: Migrate to Qodo Merge AI Review (Gemini 3 Flash)
  ([`cdbd76b`](https://github.com/n24q02m/qwen3-embed/commit/cdbd76b9c0ca8fa71a5c9e53320187bff47112bf))

### Testing

- Add comprehensive test coverage for utilities and components
  ([`f1d2997`](https://github.com/n24q02m/qwen3-embed/commit/f1d2997494ce85dedd9c8a537c779e3793bfff11))


## v1.1.3 (2026-02-18)

### Bug Fixes

- Correct model identifiers from Qwen/ to n24q02m/ namespace
  ([`298405c`](https://github.com/n24q02m/qwen3-embed/commit/298405c49cf7a9e43c532b7f9a17d79679d1a3c0))


## v1.1.2 (2026-02-18)

### Bug Fixes

- GGUF Reranker Device.AUTO defaulting to CPU instead of GPU
  ([`684af12`](https://github.com/n24q02m/qwen3-embed/commit/684af122e3ae9defa76ce73b245abc7d54389b16))

### Chores

- Update uv.lock
  ([`200a611`](https://github.com/n24q02m/qwen3-embed/commit/200a61171036cbd93f4a17b86c01a37d94c1417b))


## v1.1.1 (2026-02-17)

### Bug Fixes

- Auto-detect GPU for GGUF backend (Device.AUTO uses n_gpu=-1)
  ([`edb58bb`](https://github.com/n24q02m/qwen3-embed/commit/edb58bb9d91153fef4c282ee365af517f750d67b))

### Documentation

- Add GPU acceleration section and fix CPU-only claim
  ([`add0204`](https://github.com/n24q02m/qwen3-embed/commit/add0204b7eb0d7712eddf91589e9cfda304fa8ba))


## v1.1.0 (2026-02-17)

### Chores

- Standardize mise.toml (add node, ty check, UV_LINK_MODE, settings)
  ([`4ba23ef`](https://github.com/n24q02m/qwen3-embed/commit/4ba23efb9c2e66015330de87bb08229fbc3264a4))

### Features

- Auto-detect DirectML GPU provider and improve logging
  ([`f400012`](https://github.com/n24q02m/qwen3-embed/commit/f400012168e320b9403327be238922f9be648661))


## v1.0.0 (2026-02-14)

### Bug Fixes

- **cd**: Remove build_command from PSR config (not available in PSR container)
  ([`d63c86b`](https://github.com/n24q02m/qwen3-embed/commit/d63c86b2153c5285cd93f456f927ea23cba62a1e))

### Chores

- Migrate from release-please to python-semantic-release v10
  ([`9a78411`](https://github.com/n24q02m/qwen3-embed/commit/9a78411094c12f7c83c6a0a14e552927757c0df4))

- Sync beta manifest from stable [skip ci]
  ([`40d9c6d`](https://github.com/n24q02m/qwen3-embed/commit/40d9c6d049e349e58f1a8daee35ed8e85c26f932))


## v0.2.1 (2026-02-14)

### Chores

- Sync beta manifest from stable [skip ci]
  ([`a8b9c84`](https://github.com/n24q02m/qwen3-embed/commit/a8b9c84d2b98f2259090bf7ba664563c203ad649))

- **dev**: Release 0.2.1-beta ([#5](https://github.com/n24q02m/qwen3-embed/pull/5),
  [`8d8dbc1`](https://github.com/n24q02m/qwen3-embed/commit/8d8dbc12ce60aec6b68b3c85ca5146b43de18f80))

- **main**: Release 0.2.1 ([#6](https://github.com/n24q02m/qwen3-embed/pull/6),
  [`7815ee6`](https://github.com/n24q02m/qwen3-embed/commit/7815ee6fe2b061b60e6d9b1848411d9a5e763e72))

### Continuous Integration

- Allow GHSA-w8v5-vhqr-4h9v (diskcache, no patch available)
  ([#5](https://github.com/n24q02m/qwen3-embed/pull/5),
  [`8d8dbc1`](https://github.com/n24q02m/qwen3-embed/commit/8d8dbc12ce60aec6b68b3c85ca5146b43de18f80))

### Documentation

- Update README with Q4F16 and GGUF variants ([#5](https://github.com/n24q02m/qwen3-embed/pull/5),
  [`8d8dbc1`](https://github.com/n24q02m/qwen3-embed/commit/8d8dbc12ce60aec6b68b3c85ca5146b43de18f80))

### Features

- Add Q4F16 ONNX and GGUF model variant support
  ([#5](https://github.com/n24q02m/qwen3-embed/pull/5),
  [`8d8dbc1`](https://github.com/n24q02m/qwen3-embed/commit/8d8dbc12ce60aec6b68b3c85ca5146b43de18f80))

- Promote dev to main (v0.2.1-beta) ([#5](https://github.com/n24q02m/qwen3-embed/pull/5),
  [`8d8dbc1`](https://github.com/n24q02m/qwen3-embed/commit/8d8dbc12ce60aec6b68b3c85ca5146b43de18f80))

### Refactoring

- Use dedicated *-GGUF HF repos for GGUF models
  ([#5](https://github.com/n24q02m/qwen3-embed/pull/5),
  [`8d8dbc1`](https://github.com/n24q02m/qwen3-embed/commit/8d8dbc12ce60aec6b68b3c85ca5146b43de18f80))


## v0.2.0 (2026-02-14)

- Initial Release
