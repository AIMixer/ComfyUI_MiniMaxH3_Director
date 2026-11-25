# 现成视频当某一段（video-as-segment）与「空选择＝拼接」

> 本文是实现说明，配套 PR：`feat: 支持「现成视频当某一段」，并在空选择时直接拼接`。
> 面向 review，尽量写清"为什么这样改"以及踩过的坑。

---

## 一、要解决的问题

批量（`prompt_batch` / r2v）时间线上，"这一段我已经有成品视频了"是常见情况：

- 上一条时间线已经生成好、并且验收过的段；
- 另一个项目/工具产出的片段；
- 只改了后面几段，前面的不想重跑。

以前只有一条路：**整条时间线重跑**。哪怕只想改第 7 段，前面 6 段也要重新采样。

---

## 二、为什么这其实只是个「播种」问题

导演台里"把现成内容当作一段，后面的段从它续写"的两个半截**本来就有**：

**1. 命中缓存＝不采样。** `executor_core` 里：

```python
cached = load_segment_cache(node_id, seg, plan)
if cached is None:
    cached = load_segment_cache(node_id, seg, plan, allow_stale=True)   # 过期也认
if cached is not None:
    completed_outputs[seg.index] = cached          # 像素：直接作为该段输出
    completed_av_latents[seg.index] = cached_av    # latent：供后面段续写
```

> 注意 `allow_stale=True`：`segment_cache.py` 的注释也写明了
> *"Gen timelines (no source clips) never count as a source change, so stale
> fill/continuity still work after pipeline-only fingerprint churn."*
> 也就是说 r2v 这种没有源片的"gen 时间线"，提示词/参考图变化不会让旧缓存作废。

**2. 上一段的 AV latent 会传给下一段。** `h3_latent_continue.apply_latent_continue`
把已采样段的音视频 latent 尾部写进下一段的空 latent，并建 Nose mask。

**缺的只是"用视频把缓存填上"这一步。** 所以新增的 `director/seed_from_video.py` 只做一件事：

```
解码视频 → 视频 VAE 编码 → 音频 VAE 编码 → save_segment_cache(...)
```

之后全部沿用既有机制，没有改动采样、没有改动导出。

---

## 三、`runSelection` 的语义（一个容易搞反的地方）

`runSelection` 是**"要生成哪几段"**，不是"要保留哪几段"。

- `run_indices is None` → **全跑**（每段都采样）
- `run_indices = frozenset({...})` → **只采样这几段**，其余从缓存取

所以"拼接模式"要的是**空集合**，而不是"全选"。`plan.py` 里原本：

```python
if not indices:
    raise ValueError("「选择运行」已开启但未勾选任何片段/提示词组。请至少勾选一组再执行。")
```

这条在拼接模式下必须放行：

```python
if not indices:
    if timeline.get("assembleOnly") or timeline.get("assemble_only"):
        log.info("Assemble mode: empty run selection is intentional -> no segment will be sampled")
        return frozenset()
    raise ValueError(...)
```

**保留非空选择的原行为**很重要：勾了段就照常重跑 —— 这也正是"某段换了源视频之后刷新它的种子"的途径。

---

## 四、开销：只在必要时才编码 latent

latent 的用途只有"给**后面要被采样**的段做续写种子"。

- 拼接模式：没有任何段被采样 → **一段 latent 都不用编** → 跳过 VAE 编码（日志 `pixels only`）
- 有后续段要采样时：才编码

实测（1056×608，2 段 × 192 帧）：

| 模式 | 耗时 |
|---|---|
| 全采样 | ~380 s |
| 拼接（跳过 VAE 编码） | **30 s** |

---

## 五、踩过的坑（改这个插件时值得知道）

**1. `SegmentPlan(` 有 6 个构造点，只改 `plan.py` 不生效。**

```
director/external_groups.py:741, :812
director/fl2v_timeline.py:681
director/gen_timeline.py:681      ← prompt_batch 实际走的是这条
director/plan.py:939
director/segment_cache.py:1170
```

第一次只改了 `plan.py`，结果计划里 `source_video` 始终为空、播种静默跳过。

**2. 播种只在 `nodes/director.py` 的 `execute` 里做是不够的。**

"空选择就报错"发生在 `prepare_director_plan` 阶段，**比播种钩子更早**。
所以拼接模式判定必须在建计划**之前**先改 `timeline_data`。

**3. 这个仓库里各家文件行尾不统一。**
`director/plan.py` 是 CRLF，其余多数是 LF。一律归一化会让 diff 从 32 行膨胀到
2554 行，PR 就没法看了。生成补丁要**按各文件原本的行尾对齐**。

**4. ComfyUI 对自定义节点不做热重载。**
改完必须重启；且 `custom_nodes/` 下的备份目录**会被当成插件扫描**并报
`IMPORT FAILED`，备份要放到 `custom_nodes/` 之外。

---

## 六、界面

素材组卡片中，紧跟在 Video 区之下：

```
VIDEOS                                    From library   0/3
[Video 1] [Video 2] [Video 3]
──────────────────────────────────────────────────────────
现成视频当这一段 / Finished video AS this clip
                            From library   未设置
```

- 未设置：`未设置：本段正常生成`（行为零变化）
- 已设置：`✓ 本段直接采用：xxx.mp4（不采样；后续段从它续写）` ＋ **清除 / Clear**

复用了现有的 `createR2vSection(...)` 区块工厂与 `editor.chooseVideoInput()` 选择器。

---

## 七、测试记录

**播种**：一段由导出的 `seg_0002.mp4` 播种、紧随其后的段**正常采样**并从它续写；
产物逐帧核对，第一段内容与源视频一致。日志：

```
Seed-from-video: segment 1/2 seeded from seg_0002.mp4 (192 frames, +av_latent)
```

**拼接**：两段均有源视频、选择留空。日志：

```
Assemble mode (empty run selection): 2 segment(s), nothing selected -> concatenating, no sampling
Seed-from-video: segment 1 needs no continuity seed -> skipping VAE encode
Seed-from-video: segment 1/2 seeded from seg_0002.mp4 (192 frames, pixels only)
Merged 2 segment audio clip(s) for export=all (384 frames)
```

产物 384 帧 ＝ 两段首尾相接，端到端 30 秒。

**补丁**：在 `4c6245e` 与 `a8f57b8` 上均 `git apply --check` 通过，套用结果与工作树逐字节一致。

---

## 八、待商榷

- `minimax_timeline.js` 里那两个区块只在非批量的段视图/全局视图渲染，批量卡片才是主入口。
  保留两份是为了视图一致，若认为冗余可以只留 `minimax_image_batch.js` 那份。
- 标签目前是硬编码中英双语，未进 `minimax_i18n.js`。需要的话改成 i18n 键。
- `assemble_only` 从时间线字典读取，未做成 `DirectorPlan` 字段（空 `run_indices` 已能覆盖判断）。
- `sourceVideo` 目前只能通过时间线 JSON 或 UI 设置，没有做成节点输入（`MMX_DIR_GROUP` 那条路已有
  `ref_videos`，语义不同：那是"给模型看的参考视频"，不是"直接当成品"）。
