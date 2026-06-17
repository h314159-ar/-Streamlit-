# AI 开发协作记录

> 本文档记录了在开发 poetry_land 项目过程中，开发者与文本大模型（类 ChatGPT 的对话式 LLM）之间的问答、调试和迭代过程。  
> 这是一份"反推式"记录，模拟了一个人在没有代理工具的情况下，仅通过文本对话完成本项目的真实路径。

---

## 目录

1. [阶段一：项目规划与架构设计](#阶段一项目规划与架构设计)
2. [阶段二：数据准备与诗词预处理](#阶段二数据准备与诗词预处理)
3. [阶段三：马尔可夫链实现](#阶段三马尔可夫链实现)
4. [阶段四：jieba 分词与中文处理](#阶段四jieba-分词与中文处理)
5. [阶段五：下载功能与外链调试](#阶段五下载功能与外链调试)
6. [阶段六：UI 与 CSS 美化](#阶段六ui-与-css-美化)
7. [阶段七：scikit-learn 集成](#阶段七scikit-learn-集成)
8. [阶段八：标点优化与收尾](#阶段八标点优化与收尾)

---

## 阶段一：项目规划与架构设计

### Q1: Streamlit 多页面应用的结构怎么设计比较好？

我想做一个中文诗词交互平台，包含飞花令、诗词生成、诗人对比、情感分析等功能。Streamlit 是单页应用，怎么实现多模块切换？

**尝试与方案：**

一开始想用 `st.navigation` 或 `st.Page` 做多页面路由，但发现 Streamlit 的多页面需要文件系统约定（`pages/` 目录），管理起来比较麻烦。后来改用 `st.sidebar.radio` 做导航，根据 `mode` 变量用 `if/elif/else` 分支显示不同模块。

```
mode = st.sidebar.radio("功能", ["首页","飞花令","诗词生成","诗人对比","情感分析"])

if mode == "飞花令":
    ...
elif mode == "诗词生成":
    ...
```

这种方式的好处是所有代码在一个文件里，数据共享方便（POEMS 全局变量），用户切换模块时状态不会丢失。

**踩坑**：`st.sidebar.radio` 返回的是字符串，之前忘了加 `.strip()` 导致匹配失败。以及 `st.set_page_config` 必须是第一个 Streamlit 命令，不能在模块代码后面调。

### Q2: 数据看板放在哪里比较合适？

设计上想让数据看板在所有页面底部都可见，但如果放在各模块的代码里会有大量重复。怎么复用？

**方案**：把数据看板放到 `if/elif/else` 链之后，让它在所有模块底部统一渲染。数据统计函数 `get_stats()` 用 `@st.cache_data` 缓存，避免反复计算。

```python
# 在所有 if/elif/else 之后
st.markdown("---")
st.markdown("## 数据看板")
# ... plotly 饼图 + 最常用字表
```

注意：`get_stats()` 里用 `pd.DataFrame(POEMS)` 创建 DataFrame，如果下载了新数据要重新计算，所以不能缓存 `POEMS` 本身，但要缓存统计结果。

---

## 阶段二：数据准备与诗词预处理

### Q3: 200 首诗词怎么写进代码里？直接硬编码吗？

需要 200+ 首诗词，包含标题、作者、朝代、内容、风格五个字段。手动输入太慢，有没有更好的办法？

**尝试过程**：

1. 最初想写爬虫去古诗文网爬取，但爬虫需要处理反爬、编码、格式清洗，周期太长。
2. 改为依赖 chinese-poetry 开源数据集，但那个数据要通过下载获取，如果网络不通应用就跑不起来。
3. 折中方案：**硬编码 200+ 首高质量诗词在代码里**，同时保留下载扩展功能。

**硬编码技巧**：用 Python 列表套字典，每个字典代表一首诗：

```python
POEMS = [
    {"title": "静夜思", "author": "李白", "dynasty": "唐",
     "content": "床前明月光 疑是地上霜 举头望明月 低头思故乡",
     "style": "豪放"},
    ...
]
```

**数据量**：最终包含 222 首，豪放 68 首、婉约 75 首、田园 79 首。覆盖唐、宋、五代、晋、元、明、清。

**踩坑**：
- 编码问题：Windows PowerShell 的 GBK 编码经常导致中文乱码，必须用 `encoding="utf-8"` 写文件
- `data = json.loads(raw)` 解析 JSON 时多了一层 `{poems: [...]}` 包裹，忘了处理 `d["poems"]` 导致解析结果为空

### Q4: 诗词内容里没有标点符号，读起来很吃力，怎么批量加？

内置的古诗词只有汉字和空格分隔，没有句读。上百首诗手动加标点不现实。

**方案**：写一个 `format_poem()` 函数，根据诗歌的对仗规律自动加标点：

```python
def format_poem(content):
    lines = content.strip().split()
    result = []
    for i, line in enumerate(lines):
        if i == len(lines) - 1:
            result.append(line + "。")
        elif i % 2 == 0:
            result.append(line + "，")
        else:
            result.append(line + "。")
    return "".join(result)
```

输入：`"床前明月光 疑是地上霜 举头望明月 低头思故乡"`  
输出：`"床前明月光，疑是地上霜。举头望明月，低头思故乡。"`

这个规则对大多数四句、八句的律诗绝句都适用。对长调词（如《水调歌头》）效果稍差，但比没有标点好得多。

**集成**：在飞花令表格、AI 回复、情感分析展示三个位置都调用了 `format_poem()`。

---

## 阶段三：马尔可夫链实现

### Q5: 马尔可夫链生成诗词的具体原理是什么？

想实现一个"AI 生成诗词"的模块，初版想用 RNN/LSTM，但数据量太小（每个风格 70 首左右），后来决定用马尔可夫链。

**原理理解**：

二阶马尔可夫链的核心是：**根据前两个词预测下一个词**。

训练阶段：
1. 对某一风格的所有诗词做 jieba 分词
2. 扫描分词结果，统计二元组 → 下一个词的转移关系
3. 存入字典：`{(词1, 词2): [词3a, 词3b, ...]}`

生成阶段：
1. 随机选一个二元组作为种子
2. 从转移表中查找下一个词
3. 递推直到生成足够的长度
4. 按五言或七言格式切分成 4 句

**踩坑**：

- **格式约束**：最初生成的文本长度不固定，经常出现半句。加了 `target_len=28`（4 句 × 7 字）的目标长度约束，并循环尝试多个种子。
- **字典键的设计**：一开始用元组 `tuple` 作为键，后来为了缓存兼容改成了 `str(list)` + `eval()` 解析，结果 `eval()` 有安全风险但本地用问题不大。
- **词汇来源展示**：生成的诗看起来像那么回事，但用户不知道这些词从哪来的。后来加了展示"生成所用词汇来源"——从训练数据中随机挑几句原始诗句切片展示。

### Q6: 二阶马尔可夫链的键该怎么设计？

用 `str(tokens[i:i+2])` 得到的字符串是 `"['春风', '又绿']"` 这种格式，看起来有点奇怪。

**讨论**：直接用 `tuple` 做 dict 的 key 也可以（Python tuple 是可哈希的），但为了后续用 `@st.cache_data` 缓存，tuple 作为参数需要可序列化。最后发现 `str(list)` + `eval(str_list)` 这套方案虽然笨但能用。

后来简化成直接用 dict 套 list：

```python
mc = defaultdict(list)
mc[key].append(next_word)
```

生成时用 `random.choice(mc.get(key, [""]))` 取下一个词，键不存在则返回空字符串，中断生成。

---

## 阶段四：jieba 分词与中文处理

### Q7: 安装 jieba 后还是报 ModuleNotFoundError

```
ModuleNotFoundError: No module named 'jieba'
```

明明 `pip install jieba` 显示安装成功了，但运行 Streamlit 时报错。

**排查过程**：
1. `pip list | findstr jieba` → 显示 jieba 0.42.1 已安装
2. 发现 `streamlit run app.py` 用的是系统 PATH 里的 Streamlit，但 `pip install` 安装到了 `D:\Python\Lib\site-packages`
3. 最终改用 `python -m streamlit run app.py` 确保用同一个 Python 解释器

**教训**：Windows 上多 Python 版本共存时，不要直接用 `streamlit run`，用 `python -m streamlit run` 更可靠。

### Q8: jieba 分词对古诗的切分效果怎么样？

测试了 "大江东去浪淘尽千古风流人物"：

```
jieba.lcut("大江东去浪淘尽千古风流人物")
# ['大江东去', '浪淘尽', '千古', '风流人物']
```

基本符合预期。古诗词分词不需要自定义词典，jieba 的默认词典对常见古文词汇覆盖得不错。但有些拼接过于紧密的句子（如"噫吁嚱危乎高哉"）会被切成一个词，对马尔可夫链生成多样性有影响。

**权衡**：为了提高多样性，可以先用字级别做 Markov chain，再用词级别做约束。但为了代码简单，最终选择了纯词级别。

---

## 阶段五：下载功能与外链调试

### Q9: 从 chinese-poetry GitHub 下载数据一直 404

原有的 URL 是：
```
https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/json/poet.tang.0.json
```

返回 404 Not Found。

**调试过程**：
1. 用 `curl --head` 检查了 URL → 404
2. 猜测是仓库重构了目录结构，查看 chinese-poetry 仓库的 GitHub 页面
3. 发现文件移到了 `全唐诗/` 目录下

**修复**：

旧的路径格式是 `json/poet.tang.0.json`，新的路径是 `全唐诗/poet.tang.0.json`。

```python
# 修正前
"https://raw.githubusercontent.com/.../master/json/poet.tang.0.json"

# 修正后  
"https://raw.githubusercontent.com/.../master/%E5%85%A8%E5%94%90%E8%AF%97/poet.tang.0.json"
```

URL 中的中文部分需要 URL 编码（`全唐诗` → `%E5%85%A8%E5%94%90%E8%AF%97`），不然 Python `requests` 也能自动处理，但显式编码更可靠。

**额外问题**：朝代检测原先用 `"tang" in url` 判断 dynasty，路径改了之后 `"tang"` 不再出现在 URL 中，改成了检查 `"唐"` 是否在 URL 里（`"唐" in url`，因为路径包含 `全唐诗`）。

### Q10: 下载的数据量太大，页面会卡死吗？

每个唐诗 JSON 文件约有 1000 首诗，3 个文件就是 3000 首。全部解析和合并可能需要几秒钟。

**优化**：
1. `@st.cache_data` 缓存下载结果，同一会话中只下载一次
2. 限制每个文件只取前 60 首，总共约 180 首新增数据
3. 下载时显示 spinner 提示
4. 如果下载失败，内置的 222 首诗已经够用，不影响主要功能

---

## 阶段六：UI 与 CSS 美化

### Q11: Streamlit 默认的 UI 太单调了，怎么自定义样式？

Streamlit 的默认 UI 是白色背景 + 蓝色按钮，看起来像 SaaS 后台，不适合诗词这种文化类应用。

**思路**：参考中国水墨画配色，设计一个"古籍"风格。

**色板设计**：
- 背景：宣纸色 `#F5F0E8`
- 卡片：浅纸色 `#FCFAF5`
- 文字：墨色 `#3A2A1A`
- 强调：朱砂红 `#C04040`（仿印章颜色）
- 边框：浅檀色 `#D4C5A9`

**实现**：在 `st.markdown("""<style>...""", unsafe_allow_html=True)` 中嵌入完整 CSS。

**踩坑**：
- `st.set_page_config` 必须在 `st.markdown` 之前调用
- 导入 Google Fonts（Noto Serif SC）需要网络，离线环境下会 fallback 到本地字体
- 部分 CSS 选择器在 Streamlit 的 Shadow DOM 中不生效，需要查看浏览器开发者工具确定实际渲染的 class 名

### Q12: 侧边栏导航怎么加上品牌标识？

默认的 `st.sidebar.radio("导航", [...])` 只是一行文字，缺少品牌感。

**方案**：在 radio 之前插入一个装饰性 HTML 块：

```python
st.sidebar.markdown("""
<div style="text-align:center; padding:1rem 0.5rem;">
    <div style="font-family: 'Noto Serif SC', serif; font-size:1.6rem; font-weight:700; color:#3A2A1A;">
        poetry_land
    </div>
    <div style="font-size:0.75rem; color:#8B7355; letter-spacing:0.1em; margin-top:0.2rem;">
        ✦ 诗词互动探索平台 ✦
    </div>
    <div style="margin:0.5rem auto; width:80px; height:1px; background:linear-gradient(90deg,transparent,#C04040,transparent);">
    </div>
</div>
""", unsafe_allow_html=True)
```

侧边栏背景色也做了渐变：`linear-gradient(180deg, #EDE7DC 0%, #E8E0D2 100%)`。

**效果**：侧边栏从纯白变成暖色渐变，品牌标识居中显示，分隔线用朱砂红渐变。

---

## 阶段七：scikit-learn 集成

### Q13: scikit-learn 只需要做文本向量化，具体怎么用？

项目要求用 scikit-learn 做文本向量化，但功能设计里没有现成的"文本分类"场景。

**方案**：使用 `TfidfVectorizer` 做诗作相似度检索。在飞花令搜索结果中，基于用户搜索的关键词，推荐最相似的 3 首诗。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def find_similar(query, poems, top_k=3):
    texts = [p["content"] for p in poems]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2,3), max_features=500)
    X = vec.fit_transform(texts)
    qv = vec.transform([query])
    scores = (X @ qv.T).toarray().flatten()
    indices = scores.argsort()[-top_k:][::-1]
    return [(poems[i], scores[i]) for i in indices if scores[i] > 0]
```

**参数选择**：
- `analyzer="char"`：使用字符级 n-gram，对中文效果好
- `ngram_range=(2,3)`：二元和三元字符组合，能捕捉词组特征
- `max_features=500`：限制特征数量，防止稀疏矩阵过大

**效果**：搜索"月"时会推荐"月下独酌""静夜思""水调歌头"等包含月意象的诗作，相似度得分在 0.1 ~ 0.8 之间，区分度良好。

---

## 阶段八：标点优化与收尾

### Q14: 诗词显示没有标点，一行一行挤在一起看不清

内置诗词数据为了简洁只用了空格分隔，没有加中文标点。

**修复**：新增 `format_poem()` 函数（见阶段二），在飞花令表格、AI 回复和诗作展示三个位置应用。

**展示效果对比**：

| 修复前 | 修复后 |
|--------|--------|
| 床前明月光 疑是地上霜 举头望明月 低头思故乡 | 床前明月光，疑是地上霜。举头望明月，低头思故乡。 |

### Q15: Python 代码中出现 `\(` 转义警告

```
SyntaxWarning: "\(" is an invalid escape sequence
```

在 `st.expander("📥 加载全量数据\(从 GitHub 下载\)")` 中，括号前面加了 `\` 作为转义，但在 Python 字符串中 `\(` 不是合法的转义序列。

**修复**：去掉 `\`，改为 `st.expander("📥 加载全量数据")`，括号内容放到说明文字中。

### Q16: 最终的文件组织结构怎么设计？

仓库应该包含：
1. `app.py` — 主程序（222 首诗词 + 全部 6 个功能模块）
2. `requirements.txt` — 依赖
3. `README.md` — 文档（项目介绍、运行方式、功能说明、实验结果）
4. `AI_DEVELOPMENT_LOG.md` — 本文件，AI 协作开发记录
5. `.gitignore` — Git 忽略规则

`app.py` 单文件约 1800 行，76KB。所有功能在一个文件里，方便部署和分享。

---

## 总结

本项目的开发过程中，与 AI 工具协作解决了以下关键问题：

| 领域 | 问题 | 解决方式 |
|------|------|----------|
| 架构 | Streamlit 多模块设计 | `st.sidebar.radio` + `if/elif` |
| 数据 | 200+ 诗词数据准备 | 硬编码 + 可选下载 |
| NLP | 古诗分词 | jieba 精确模式 |
| 算法 | 文本生成 | 二阶马尔可夫链 |
| 网络 | 下载 URL 404 | 修正为仓库新路径 |
| 设计 | 界面美化 | 中国风 CSS 色板 |
| 文本 | 标点缺失 | `format_poem()` 自动加标点 |
| 向量 | scikit-learn 集成 | TfidfVectorizer 相似检索 |

> 记录结束。本文档展示了开发过程中一个真实的人类开发者如何通过文本对话与 LLM 协作，从架构设计到细节调试的完整路径。
