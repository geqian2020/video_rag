# 六道原召回失败题复核

2026-10-23 恢复记录：从仍保留的原始四视频 details.jsonl 重新整理。以下为 agent 建议，人工审核仍为 0；未修改旧标签或旧成绩。

## 原记录 23：What must stay the same when continuing an earlier graph conversation?

原标签：7WC12HHC55E，2422.079–2444.800 秒。

语义不匹配：返回讨论状态保留，没有直接回答 same thread ID；暂保留失败。

| 原重排名次 | 返回来源 | 时间（秒） |
|---|---|---|
| 1 | [7WC12HHC55E](https://www.youtube.com/watch?v=7WC12HHC55E&t=2294s) | 2294.960–2310.960 |
| 2 | [7WC12HHC55E](https://www.youtube.com/watch?v=7WC12HHC55E&t=2560s) | 2560.720–2584.240 |
| 3 | [xzRJOaqDMCg](https://www.youtube.com/watch?v=xzRJOaqDMCg&t=1319s) | 1319.840–1347.120 |

## 原记录 32：What model behavior makes precise prompts useful when building an agent?

原标签：s-jYxgKMqRc，620.998–655.728 秒。

边界与过宽标签：首名已含 precise prompting / instructions，后续 not deviate 被切断；标签还跨入工具话题，应人工精简。

| 原重排名次 | 返回来源 | 时间（秒） |
|---|---|---|
| 1 | [s-jYxgKMqRc](https://www.youtube.com/watch?v=s-jYxgKMqRc&t=608s) | 608.903–636.528 |
| 2 | [xzRJOaqDMCg](https://www.youtube.com/watch?v=xzRJOaqDMCg&t=5853s) | 5853.520–5878.639 |
| 3 | [xzRJOaqDMCg](https://www.youtube.com/watch?v=xzRJOaqDMCg&t=4598s) | 4598.400–4621.120 |

## 原记录 36：What lets a CrewAI agent take action beyond the language model itself?

原标签：s-jYxgKMqRc，1811.703–1850.408 秒。

证据充分性待查：出现 tools 不等于已说明可在 LLM 之外行动，不能据关键词直接改为命中。

| 原重排名次 | 返回来源 | 时间（秒） |
|---|---|---|
| 1 | [s-jYxgKMqRc](https://www.youtube.com/watch?v=s-jYxgKMqRc&t=859s) | 859.633–919.103 |
| 2 | [kBXYFaZ0EN0](https://www.youtube.com/watch?v=kBXYFaZ0EN0&t=585s) | 585.440–643.880 |
| 3 | [kBXYFaZ0EN0](https://www.youtube.com/watch?v=kBXYFaZ0EN0&t=2270s) | 2270.640–2308.800 |

## 原记录 40：Why does the presenter expect developers to maintain their own tool library?

原标签：s-jYxgKMqRc，2757.268–2794.458 秒。

边界及替代来源：own library 的示例被切断，其他返回谈内部 API 和精确操作，需人工判断是否充分。

| 原重排名次 | 返回来源 | 时间（秒） |
|---|---|---|
| 1 | [s-jYxgKMqRc](https://www.youtube.com/watch?v=s-jYxgKMqRc&t=1930s) | 1930.773–1950.778 |
| 2 | [s-jYxgKMqRc](https://www.youtube.com/watch?v=s-jYxgKMqRc&t=2742s) | 2742.093–2772.118 |
| 3 | [s-jYxgKMqRc](https://www.youtube.com/watch?v=s-jYxgKMqRc&t=1971s) | 1971.338–1993.768 |

## 原记录 48：How are ordinary functions made available as tools in this example?

原标签：kBXYFaZ0EN0，1693.559–1721.720 秒。

跨视频替代证据：首名另一个视频明确提及 LangChain tool decorator；this example 来源不明，应补替代标签或限定问题。

| 原重排名次 | 返回来源 | 时间（秒） |
|---|---|---|
| 1 | [s-jYxgKMqRc](https://www.youtube.com/watch?v=s-jYxgKMqRc&t=2678s) | 2678.957–2724.625 |
| 2 | [s-jYxgKMqRc](https://www.youtube.com/watch?v=s-jYxgKMqRc&t=1994s) | 1994.028–2020.923 |
| 3 | [kBXYFaZ0EN0](https://www.youtube.com/watch?v=kBXYFaZ0EN0&t=1944s) | 1944.159–1977.440 |

## 原记录 55：What external capability does the EXA tool set give the agents?

原标签：kBXYFaZ0EN0，296.600–322.759 秒。

同视频漏标替代位置：第三名谈 EXA、API、search the web，需人工确认并加入替代时间段。

| 原重排名次 | 返回来源 | 时间（秒） |
|---|---|---|
| 1 | [kBXYFaZ0EN0](https://www.youtube.com/watch?v=kBXYFaZ0EN0&t=1557s) | 1557.640–1582.520 |
| 2 | [kBXYFaZ0EN0](https://www.youtube.com/watch?v=kBXYFaZ0EN0&t=3206s) | 3206.559–3235.880 |
| 3 | [kBXYFaZ0EN0](https://www.youtube.com/watch?v=kBXYFaZ0EN0&t=1424s) | 1424.520–1481.760 |

补等价证据时使用 evidence_policy=alternatives；多段缺一不可时使用 all（默认）。修订 gold 另存新版本，配对工具拒绝将不同标签混比。
