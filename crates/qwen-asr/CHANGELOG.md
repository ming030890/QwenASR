# Changelog

## [0.8.0](https://github.com/ming030890/QwenASR/compare/qwen-asr-v0.7.1...qwen-asr-v0.8.0) (2026-10-24)


### Features

* add missing parameter to qwen asr offline model ([f56e8b1](https://github.com/ming030890/QwenASR/commit/f56e8b1e58731344fad92a7ed38c59a9f09267f6))
* add missing parameter to qwen asr offline model ([6d1e38d](https://github.com/ming030890/QwenASR/commit/6d1e38da19cbae46c2afe2e1af03a5d437679ef8))
* **android:** implement K4 INT8 SDOT packed matmat decoder prefill ([b533764](https://github.com/ming030890/QwenASR/commit/b5337640c59c675f24618019030f9a04f0609970))
* **android:** implement zero-copy mmap INT8 decoder weights with pre-fused gate+up projections (Q2B) ([adc26cc](https://github.com/ming030890/QwenASR/commit/adc26ccc4be1e41cb64ae819269856fa632dbc62))
* **android:** persist VAD segment debug metadata ([bbb8937](https://github.com/ming030890/QwenASR/commit/bbb8937bce88f2cdcc0c9a714f002356db820775))
* **asr:** add offline-packed KleidiAI inference ([ad7be66](https://github.com/ming030890/QwenASR/commit/ad7be668b60ec467a6f013ea7a841b350b0dbaf9))
* **asr:** optimize quantized inference latency ([18391d6](https://github.com/ming030890/QwenASR/commit/18391d69addf19f8fffa8e124856740263c3dfc2))
* **asr:** reduce Qwen INT8 inference memory ([fe1d46a](https://github.com/ming030890/QwenASR/commit/fe1d46a8b50d25ff80c47b89ca95478babdc54f5))
* **asr:** support packed-only Android models ([2b3fc34](https://github.com/ming030890/QwenASR/commit/2b3fc340b061d5afe669d19255fc3c0250a22a3a))
* document benchmarked ASR performance ([f3b33a0](https://github.com/ming030890/QwenASR/commit/f3b33a0863c815fedbdac18650e2b673af53efbe))
* drop duplicate BF16 token embeddings and implement on-the-fly INT8 dequantization ([219d645](https://github.com/ming030890/QwenASR/commit/219d645dbf903bbbe7c36d2a786b0c675bff1c73))
* expose streaming C API for native integration ([a58befe](https://github.com/ming030890/QwenASR/commit/a58befe991393405867ece18621d60ad5dc7cfd6))
* expose streaming C API for native macOS/iOS integration ([51dc917](https://github.com/ming030890/QwenASR/commit/51dc917fc214a38ab6a85db55e2cc53ecdfabb6d))
* improve ASR WER and streaming performance ([c41d2fc](https://github.com/ming030890/QwenASR/commit/c41d2fcc5ada21aa3b64920cff8166255752e464))
* improve the live stream performace on macos ([ba47230](https://github.com/ming030890/QwenASR/commit/ba47230403f897bde2486b3235cfd3f5ca24e293))
* **jni:** add jni dependency and fully implement zero-copy JNI API ([63fc2f1](https://github.com/ming030890/QwenASR/commit/63fc2f14b79bf87902d1651f067ea1ecc54c139f))
* **jni:** expose nativeSetPrompt in JNI wrapper and resolve C-API visibility struct field error ([9265faa](https://github.com/ming030890/QwenASR/commit/9265faa6c0a52f5c1b9be9775a0e2d6b37a265cd))
* **jni:** support passing save_audio_dir to startVadSession and save post-VAD audio segments ([c778832](https://github.com/ming030890/QwenASR/commit/c77883242f44d10068b92fca6f4d200ce2dfc2c0))
* optimize the speed with auto research pattern ([9c13daa](https://github.com/ming030890/QwenASR/commit/9c13daaadd964a6c2d79bed99364b26a368c305f))
* overhaul streaming ASR inference ([f158494](https://github.com/ming030890/QwenASR/commit/f158494dd12893d0245a080617f93c2da4be9c29))
* refine the stream mode ([7ac71a9](https://github.com/ming030890/QwenASR/commit/7ac71a9a4f30b13ff155ff2eb345db85bce1b91c))
* support live from blackhold for macos for qwen-asr-cli ([724ead1](https://github.com/ming030890/QwenASR/commit/724ead1fe121d0ed0a0f7ef874142f665e7d0da3))
* update readme ([cde2178](https://github.com/ming030890/QwenASR/commit/cde21787bb545e12c154045562883b9ced00514d))
* **yue-tools:** add score subcommand to analyze token-by-token ASR and pure LM log probabilities ([7dce552](https://github.com/ming030890/QwenASR/commit/7dce55250ebe13428423fac79020bf476321b950))


### Bug Fixes

* **android:** resolve Android compilation errors and warnings ([1d85c44](https://github.com/ming030890/QwenASR/commit/1d85c44dff4a7b490b04722cd16bd0ea4e1f46e4))
* **asr:** cancel replaceable previews on stop and reuse tokenizer ([fcefafe](https://github.com/ming030890/QwenASR/commit/fcefafe31d7c5fa1056ff7243810cd8ddf7bb349))
* clean up the code ([346d112](https://github.com/ming030890/QwenASR/commit/346d112595c0d93f58c54339a7f32ca6e1e648d8))
* document qwen-asr WER benchmark scripts ([e49a138](https://github.com/ming030890/QwenASR/commit/e49a138990422f0bda9d368bec7b7bf84683bd63))
* handle split UTF-8 sequences in BPE token decoding ([75c3e31](https://github.com/ming030890/QwenASR/commit/75c3e3172b33e412fdde0bee6f8009277e9bb35e))
* handle split UTF-8 sequences in BPE token decoding ([c86fae2](https://github.com/ming030890/QwenASR/commit/c86fae2918c50f5ef74097fb4ad1e97a19387a18))
* install OpenBLAS on CI to support build ([e995cad](https://github.com/ming030890/QwenASR/commit/e995cad2c18999a39e74c56c26a2d6526020fd53))
* **kernels:** avoid parallel attention calling single-threaded OpenBLAS on Android ([3efe3ff](https://github.com/ming030890/QwenASR/commit/3efe3ff57c5527546a3d879e3f67d0bd29b8d68a))
* publish 0.2.3 with tag-driven flow ([3637ec8](https://github.com/ming030890/QwenASR/commit/3637ec80f5519ecbd0a034f6c1f23f78156cd0fe))
* publish 0.2.3 with tag-driven flow ([e7bbd18](https://github.com/ming030890/QwenASR/commit/e7bbd18dc009c3bd87f32e2346c196f65c618b19))
* trigger patch release 0.2.1 for flutter ([b5785f9](https://github.com/ming030890/QwenASR/commit/b5785f9e0a6e4cab3a4796bbd1bd401876ea5926))
* update the both library readme to mention this is WIP project ([139a591](https://github.com/ming030890/QwenASR/commit/139a5915205083abc4b87fd0228ccf4c725c99c0))
* update the release flow to support PAT ([2b9be6c](https://github.com/ming030890/QwenASR/commit/2b9be6c21b7e74e51bf1d1f15e6959679db70542))


### Performance Improvements

* **affinity:** pin inference and worker threads to performance cores on Android/Linux ([86f3e11](https://github.com/ming030890/QwenASR/commit/86f3e117d54bbafefe7ddd79ac4059c9e3c89ada))
* **asr:** add sequential shard preloading and eager readahead via madvise ([62fa081](https://github.com/ming030890/QwenASR/commit/62fa08114b0d493052b816603230aecd0e824847))
* **asr:** optimize packed decoding and finalization ([2de121e](https://github.com/ming030890/QwenASR/commit/2de121e3f3f30e8b1fc19f9f50a30790495a8d21))
* **asr:** pack quantized encoder for I8MM ([62006ca](https://github.com/ming030890/QwenASR/commit/62006cab45063d0713eedc6e5af862d947d93033))
* **asr:** quantize all encoder linear weights ([68dac70](https://github.com/ming030890/QwenASR/commit/68dac70d9c19be1308a6c2fdc4bca03694d140e8))
* **asr:** reusable activation scratch buffer, thread count tuning, and logit probing ([97348a4](https://github.com/ming030890/QwenASR/commit/97348a40680fd6dbee1d71a2aac48b6f0fd7209e))
* **asr:** use I8MM packed decoder ([51f9862](https://github.com/ming030890/QwenASR/commit/51f9862e66b83c9e53610dd63af8e903ec2d20c9))
* **decoder:** zero-allocation SwiGLU activation reuse (A2) ([b9aa20b](https://github.com/ming030890/QwenASR/commit/b9aa20ba6a498d40f98c3070358945ec0a1ba4fb))
* **encoder:** convert conv_out projection once per utterance across chunks (E2) ([c067440](https://github.com/ming030890/QwenASR/commit/c0674407bf6ecab0d5803aa06a6c4c59c34ae357))
* **kernels:** multi-thread KleidiAI prefill matmul when seq_len &gt;= 16 ([6c62453](https://github.com/ming030890/QwenASR/commit/6c62453a213da9b7840677f617c547f7be4d15d1))

## [0.7.1](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.7.0...qwen-asr-v0.7.1) (2026-05-14)


### Bug Fixes

* document qwen-asr WER benchmark scripts ([e49a138](https://github.com/huanglizhuo/QwenASR/commit/e49a138990422f0bda9d368bec7b7bf84683bd63))

## [0.7.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.6.0...qwen-asr-v0.7.0) (2026-05-14)


### Features

* improve ASR WER and streaming performance ([c41d2fc](https://github.com/huanglizhuo/QwenASR/commit/c41d2fcc5ada21aa3b64920cff8166255752e464))

## [0.6.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.5.0...qwen-asr-v0.6.0) (2026-05-10)


### Features

* document benchmarked ASR performance ([f3b33a0](https://github.com/huanglizhuo/QwenASR/commit/f3b33a0863c815fedbdac18650e2b673af53efbe))

## [0.5.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.4.2...qwen-asr-v0.5.0) (2026-03-20)


### Features

* optimize the speed with auto research pattern ([9c13daa](https://github.com/huanglizhuo/QwenASR/commit/9c13daaadd964a6c2d79bed99364b26a368c305f))

## [0.4.2](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.4.1...qwen-asr-v0.4.2) (2026-03-14)


### Bug Fixes

* install OpenBLAS on CI to support build ([e995cad](https://github.com/huanglizhuo/QwenASR/commit/e995cad2c18999a39e74c56c26a2d6526020fd53))

## [0.4.1](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.4.0...qwen-asr-v0.4.1) (2026-03-14)


### Bug Fixes

* clean up the code ([346d112](https://github.com/huanglizhuo/QwenASR/commit/346d112595c0d93f58c54339a7f32ca6e1e648d8))

## [0.4.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.3.0...qwen-asr-v0.4.0) (2026-03-13)


### Features

* expose streaming C API for native integration ([a58befe](https://github.com/huanglizhuo/QwenASR/commit/a58befe991393405867ece18621d60ad5dc7cfd6))
* expose streaming C API for native macOS/iOS integration ([51dc917](https://github.com/huanglizhuo/QwenASR/commit/51dc917fc214a38ab6a85db55e2cc53ecdfabb6d))


### Bug Fixes

* handle split UTF-8 sequences in BPE token decoding ([75c3e31](https://github.com/huanglizhuo/QwenASR/commit/75c3e3172b33e412fdde0bee6f8009277e9bb35e))
* handle split UTF-8 sequences in BPE token decoding ([c86fae2](https://github.com/huanglizhuo/QwenASR/commit/c86fae2918c50f5ef74097fb4ad1e97a19387a18))

## [0.3.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.2.4...qwen-asr-v0.3.0) (2026-02-23)


### Features

* add missing parameter to qwen asr offline model ([f56e8b1](https://github.com/huanglizhuo/QwenASR/commit/f56e8b1e58731344fad92a7ed38c59a9f09267f6))
* add missing parameter to qwen asr offline model ([6d1e38d](https://github.com/huanglizhuo/QwenASR/commit/6d1e38da19cbae46c2afe2e1af03a5d437679ef8))
* improve the live stream performace on macos ([ba47230](https://github.com/huanglizhuo/QwenASR/commit/ba47230403f897bde2486b3235cfd3f5ca24e293))
* refine the stream mode ([7ac71a9](https://github.com/huanglizhuo/QwenASR/commit/7ac71a9a4f30b13ff155ff2eb345db85bce1b91c))
* support live from blackhold for macos for qwen-asr-cli ([724ead1](https://github.com/huanglizhuo/QwenASR/commit/724ead1fe121d0ed0a0f7ef874142f665e7d0da3))
* update readme ([cde2178](https://github.com/huanglizhuo/QwenASR/commit/cde21787bb545e12c154045562883b9ced00514d))


### Bug Fixes

* publish 0.2.3 with tag-driven flow ([3637ec8](https://github.com/huanglizhuo/QwenASR/commit/3637ec80f5519ecbd0a034f6c1f23f78156cd0fe))
* publish 0.2.3 with tag-driven flow ([e7bbd18](https://github.com/huanglizhuo/QwenASR/commit/e7bbd18dc009c3bd87f32e2346c196f65c618b19))
* trigger patch release 0.2.1 for flutter ([b5785f9](https://github.com/huanglizhuo/QwenASR/commit/b5785f9e0a6e4cab3a4796bbd1bd401876ea5926))
* update the both library readme to mention this is WIP project ([139a591](https://github.com/huanglizhuo/QwenASR/commit/139a5915205083abc4b87fd0228ccf4c725c99c0))
* update the release flow to support PAT ([2b9be6c](https://github.com/huanglizhuo/QwenASR/commit/2b9be6c21b7e74e51bf1d1f15e6959679db70542))

## [0.2.3](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.2.2...qwen-asr-v0.2.3) (2026-02-22)


### Bug Fixes

* update the release flow to support PAT ([2b9be6c](https://github.com/huanglizhuo/QwenASR/commit/2b9be6c21b7e74e51bf1d1f15e6959679db70542))

## [0.2.2](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.2.1...qwen-asr-v0.2.2) (2026-02-22)


### Bug Fixes

* publish 0.2.3 with tag-driven flow ([3637ec8](https://github.com/huanglizhuo/QwenASR/commit/3637ec80f5519ecbd0a034f6c1f23f78156cd0fe))
* publish 0.2.3 with tag-driven flow ([e7bbd18](https://github.com/huanglizhuo/QwenASR/commit/e7bbd18dc009c3bd87f32e2346c196f65c618b19))

## [0.2.1](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.2.0...qwen-asr-v0.2.1) (2026-02-22)


### Bug Fixes

* trigger patch release 0.2.1 for flutter ([b5785f9](https://github.com/huanglizhuo/QwenASR/commit/b5785f9e0a6e4cab3a4796bbd1bd401876ea5926))
* update the both library readme to mention this is WIP project ([139a591](https://github.com/huanglizhuo/QwenASR/commit/139a5915205083abc4b87fd0228ccf4c725c99c0))

## [0.2.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.1.2...qwen-asr-v0.2.0) (2026-02-22)


### Features

* add missing parameter to qwen asr offline model ([f56e8b1](https://github.com/huanglizhuo/QwenASR/commit/f56e8b1e58731344fad92a7ed38c59a9f09267f6))
* add missing parameter to qwen asr offline model ([6d1e38d](https://github.com/huanglizhuo/QwenASR/commit/6d1e38da19cbae46c2afe2e1af03a5d437679ef8))
