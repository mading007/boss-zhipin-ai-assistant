import json
import re
import ast
import pandas as pd
import requests
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 第一步：填写你的【个人简历画像】
# ============================================================
MY_PROFILE = {
    'tech_skills': ['Python', 'LangChain', 'Chroma', 'RAG', 'Prompt',
                    '大模型', 'API', 'Pandas', 'NumPy', 'MySQL', 'SQL',
                    '自动化', 'pytest', 'Selenium', '接口测试'],
    'years_exp': 0.0,
    'education': '本科',
    'expected_salary': 15,
    'preferred_company_size': 0,
    'personal_summary': """
我是马丁，2026届电子信息工程本科毕业生，可立即到岗，期望城市北京。

核心项目：
1. RAG论文问答系统（8月，已开源）：基于 LangChain + Chroma + BGE Embedding 独立开发，
   支持 PDF 上传、文本切分、向量化存储、语义检索与答案引用溯源，完整走通 RAG 全链路；
2. 智能求职助手（9月）：Chrome DevTools Protocol 采集岗位数据，TF-IDF + 余弦相似度做匹配排序，
   调用大模型 API 生成差异化话术，并用 HTML 面板承载，形成完整 AI 应用闭环；
3. AI图像风格统一生成（6月）：设计结构化 Prompt 模板做多维度控制，
   将风格统一度从 60% 提升至 90% 以上，沉淀 30 项一致性检查 SOP；
4. 金融终端自动化导出系统（6月）：图像识别 + 模拟操作，实现无人值守定时导出与邮件投递。

技术栈：Python、LangChain、Chroma、BGE Embedding、Prompt Engineering、
大模型 API 调用、Pandas、MySQL、自动化脚本。
""",
    'greeting_style': 'professional',
    'your_name': '马丁'
}

# ============================================================
# 第二步：配置 DeepSeek API
#    密钥存放在同目录的 .env 文件里（不要写死在代码中，也不要把 .env 提交到 Git）
#        DEEPSEEK_API_KEY=sk-xxxx
#    不配置也能运行，会降级为模板版招呼语
# ============================================================
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH, override=True)
except ImportError:
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH, 'r', encoding='utf-8') as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith('#') or '=' not in _line:
                    continue
                _k, _, _v = _line.partition('=')
                os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

DEEPSEEK_CONFIG = {
    'api_key': os.environ.get('DEEPSEEK_API_KEY', '').strip(),
    'model': os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'),
    'base_url': 'https://api.deepseek.com/v1/chat/completions'
}
# ============================================================

# ---------- 自动查找数据文件 ----------
# 猎聘列表接口不返回 JD 正文，JD 要靠单独的详情抓取补上。
# 所以这里挑「带 JD 条数最多」的那份，而不是简单地按修改时间取最新——
# 早期版本反过来排除了带 JD 的文件，结果读到 126 条全无 JD 的数据。
scripts_dir = r'D:\Desktop\boss-zhipin-scraper-master\scripts'
all_files = glob.glob(os.path.join(scripts_dir, 'liepin_api_*.json'))
if not all_files:
    raise FileNotFoundError(f"❌ 在 {scripts_dir} 下未找到 liepin_api_*.json 文件，请先运行猎聘爬虫")


def _count_jd(path):
    try:
        data = json.load(open(path, 'r', encoding='utf-8'))
    except Exception:
        return -1, 0
    if not isinstance(data, list):
        return -1, 0
    return sum(1 for x in data if str(x.get('jd', '') or '').strip()), len(data)


scored = []
for f in all_files:
    n_jd, n_all = _count_jd(f)
    scored.append((n_jd, n_all, f))
scored.sort(key=lambda x: (x[0], os.path.getmtime(x[2])), reverse=True)

latest_file = scored[0][2]
n_jd, n_all = scored[0][0], scored[0][1]
print(f"📂 数据文件: {os.path.basename(latest_file)}")
print(f"   共 {n_all} 条，其中带 JD 的 {n_jd} 条")
if len(scored) > 1:
    print("   其它候选文件：")
    for nj, na, fp in scored[1:4]:
        print(f"     {nj:3d}/{na} 带 JD  → {os.path.basename(fp)}")
if n_jd == 0:
    print("⚠️  这份数据没有 JD 正文，语义匹配会退化为兜底分。")
    print("    建议用 --detail 重新抓一遍详情。")

with open(latest_file, 'r', encoding='utf-8') as f:
    jobs = json.load(f)
df = pd.DataFrame(jobs)
print(f"📂 加载 {len(df)} 条猎聘岗位数据")

# ---------- 解析函数 ----------
# 注意：猎聘把「实习时长 + 学历」塞进了 tech_stack 字段（如 ['3个月','本科']），
# tags 字段则恒为 ' | '（空的）。所以经验/学历只能从 tech_stack 里解析，
# 而 tech_stack 不能当作技术栈来用。
EDU_LEVELS = ['大专', '本科', '硕士', '博士']


def _stack_list(row):
    stack = row.get('tech_stack', [])
    if isinstance(stack, str):
        try:
            stack = ast.literal_eval(stack)
        except Exception:
            stack = [s for s in re.split(r'[,|、\s]+', stack) if s]
    return [str(s).strip() for s in stack] if isinstance(stack, list) else []


def extract_education(row):
    """学历要求：优先 tags，其次 tech_stack"""
    for src in (row.get('tags', ''),):
        if isinstance(src, str):
            for p in src.split('|'):
                p = p.strip()
                if p in EDU_LEVELS or p == '学历不限':
                    return p
    for s in _stack_list(row):
        if s in EDU_LEVELS:
            return s
    return None


def extract_duration_months(row):
    """实习时长（月）。返回 None 表示不是按月的实习岗。"""
    for s in _stack_list(row):
        m = re.match(r'^(\d+)\s*个月$', s)
        if m:
            return int(m.group(1))
    return None


def is_internship(row):
    """猎聘上大量是实习岗，投递前需要能识别出来"""
    title = str(row.get('title', ''))
    if re.search(r'实习|见习|Intern|intern|转正', title):
        return True
    if extract_duration_months(row) is not None:
        return True
    if '提供转正' in _stack_list(row):
        return True
    return False


def parse_salary_mid(salary_str):
    """猎聘薪资格式很杂，实测以日薪为主：
       '150-200元/天'、'200元/天'、'10-15k'、'25-50万'、'2-3万/月'
       统一折算成「月薪 K」用于比较。"""
    s = str(salary_str or '')
    if not s:
        return None
    s = s.replace(' ', '')

    # 日薪 → 月薪（按 21.75 个工作日折算）
    m = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)元/天', s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2 * 21.75 / 1000
    m = re.search(r'(\d+\.?\d*)元/天', s)
    if m:
        return float(m.group(1)) * 21.75 / 1000

    # 年薪（万） → 月薪 K
    m = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)万', s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2 * 10 / 12
    m = re.search(r'(\d+\.?\d*)万', s)
    if m:
        return float(m.group(1)) * 10 / 12

    # 月薪 K（大小写 k 混用）
    m = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)[kK]', s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.search(r'(\d+\.?\d*)[kK]', s)
    if m:
        return float(m.group(1))

    # 元/月
    m = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)元/月', s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2 / 1000
    return None


df['edu_req'] = df.apply(extract_education, axis=1)
df['is_intern'] = df.apply(is_internship, axis=1)
df['salary_k'] = df.apply(lambda r: parse_salary_mid(r.get('salary', '')), axis=1)

# ---------- 中文语义匹配 ----------
# TfidfVectorizer 默认按空格切词，中文没有空格 → 整句被当成一个 token，
# 两段中文只要不是逐字相同，相似度恒为 0。所以英文按词切、中文按字符 n-gram 切。
_TFIDF_CACHE = {}


def _zh_analyzer(text):
    tokens = []
    for w in re.findall(r'[a-zA-Z][a-zA-Z0-9\+\#\.]*', str(text).lower()):
        if len(w) > 1:
            tokens.append(w)
    for chunk in re.findall(r'[\u4e00-\u9fff]+', str(text)):
        for n in (2, 3, 4):
            if len(chunk) >= n:
                tokens.extend(chunk[i:i + n] for i in range(len(chunk) - n + 1))
        if len(chunk) <= 4:
            tokens.append(chunk)
    return tokens


def semantic_score(jd_text, my_text):
    """返回 0~1 的余弦相似度；文本不足时返回 None"""
    if not jd_text or len(str(jd_text)) < 50:
        return None
    try:
        vec = _TFIDF_CACHE.get('v')
        if vec is None:
            vec = TfidfVectorizer(analyzer=_zh_analyzer, max_features=2000,
                                  sublinear_tf=True, min_df=1)
            vec.fit([my_text, str(jd_text)])
            _TFIDF_CACHE['v'] = vec
        m = vec.transform([my_text, str(jd_text)])
        return float(cosine_similarity(m[0:1], m[1:2])[0][0])
    except Exception:
        return None


def company_credibility(name):
    """猎聘上匿名/外包公司名降权"""
    s = str(name or '').strip()
    if not s:
        return 0.0, ''
    if re.search(r'^某|某(大型|中型|知名|上市|头部)?[^，,]{0,10}(公司|集团|企业|机构)|^(知名公司)$', s):
        return -8.0, '匿名/猎头'
    if re.search(r'人力资源|人才服务|猎头|劳务|外包|外派|派遣|代招', s):
        return -5.0, '人力/外包'
    return 0.0, ''


def hr_activity(status):
    """HR 活跃度：在线优先（离线时招呼语不会推送）"""
    s = str(status or '').strip()
    if not s:
        return 0.0, ''
    m = re.search(r'(\d+)\s*分钟前', s)
    if m:
        return 5.0, f'{m.group(1)}分钟前'
    m = re.search(r'(\d+)\s*小时前', s)
    if m:
        h = int(m.group(1))
        return (2.0, f'{h}小时前') if h <= 12 else (-1.0, f'{h}小时前')
    m = re.search(r'(\d+)\s*天前', s)
    if m:
        return -4.0, f'{m.group(1)}天前'
    if '刚刚' in s:
        return 5.0, '刚刚活跃'
    if '在线' in s:
        return 5.0, '在线'
    return 0.0, s[:8]


# ---------- 匹配打分 ----------
def calculate_match_score(row):
    score, details = 0.0, {}
    W_TECH, W_EXP, W_EDU, W_SALARY, W_SEMANTIC = 30, 15, 10, 10, 35

    # --- 技术栈：猎聘的 tech_stack 不是技术栈，改从 JD 正文命中关键词 ---
    jd_blob = f"{row.get('title', '')} {row.get('jd', '')}"
    my_tech = MY_PROFILE['tech_skills']
    hits = sum(1 for k in my_tech if k.lower() in jd_blob.lower())
    tech_score = min(hits / 6.0, 1.0) * W_TECH
    score += tech_score
    details['技术栈'] = round(tech_score, 1)

    # --- 经验：实习时长视为「在校生岗」，已毕业的够不上 ---
    months = extract_duration_months(row)
    if months is not None:
        exp_score = W_EXP * 0.1        # 明确要求实习期，基本是给在校生的
    elif row.get('is_intern'):
        exp_score = W_EXP * 0.2
    else:
        exp_score = W_EXP * 0.7        # 猎聘不写经验要求，给中性分
    score += exp_score
    details['经验'] = round(exp_score, 1)

    # --- 学历 ---
    edu_req = row.get('edu_req')
    edu_rank = {'大专': 1, '本科': 2, '硕士': 3, '博士': 4}
    if edu_req:
        edu_score = W_EDU if edu_rank.get(MY_PROFILE['education'], 0) >= edu_rank.get(edu_req, 0) else 0.0
    else:
        edu_score = W_EDU * 0.7
    score += edu_score
    details['学历'] = round(edu_score, 1)

    # --- 薪资（用折算后的月薪 K）---
    sal_k = row.get('salary_k')
    if sal_k and MY_PROFILE['expected_salary'] > 0:
        if sal_k >= MY_PROFILE['expected_salary']:
            salary_score = W_SALARY
        else:
            ratio = sal_k / MY_PROFILE['expected_salary']
            salary_score = W_SALARY * (1.0 if ratio >= 0.9 else 0.6 if ratio >= 0.7 else 0.2)
    else:
        salary_score = W_SALARY * 0.3
    score += salary_score
    details['薪资'] = round(salary_score, 1)

    # --- 语义 ---
    sim = semantic_score(row.get('jd', ''), MY_PROFILE.get('personal_summary', ''))
    sem_score = W_SEMANTIC * 0.3 if sim is None else min(sim * 2.2, 1.0) * W_SEMANTIC
    score += sem_score
    details['语义'] = round(sem_score, 1)

    # --- 岗位性质：实习岗直接重罚 ---
    adj = -12.0 if row.get('is_intern') else 0.0
    score += adj
    details['性质'] = adj

    # --- 公司可信度 ---
    cred_adj, cred_tag = company_credibility(row.get('boss_name', ''))
    score += cred_adj
    details['公司'] = cred_adj

    # --- HR 活跃度 ---
    act_adj, act_tag = hr_activity(row.get('boss_active_status', ''))
    score += act_adj
    details['活跃'] = act_adj

    return {
        'match_score': round(max(score, 0.0), 2),
        'detail_scores': '｜'.join(f"{k}{v}" for k, v in details.items()),
        'company_tag': cred_tag,
        'active_tag': act_tag,
    }


scored = df.apply(lambda r: pd.Series(calculate_match_score(r)), axis=1)
for _col in ['match_score', 'detail_scores', 'company_tag', 'active_tag']:
    df[_col] = scored[_col]
df_sorted = df.sort_values('match_score', ascending=False)

# ---------- 过滤硕士岗（学历分已归零，投了也是白投）----------
before = len(df_sorted)
df_sorted = df_sorted[df_sorted['edu_req'] != '硕士']
print(f"🔍 已剔除硕士岗 {before - len(df_sorted)} 条，剩余 {len(df_sorted)} 条")

_intern_n = int(df_sorted['is_intern'].sum()) if 'is_intern' in df_sorted else 0
if _intern_n:
    print(f"⚠️  其中 {_intern_n} 条是实习/转正岗（已毕业，投了大概率白投），排序中已重罚降权")

# ---------- 生成地区列 ----------
# 猎聘列表接口不返回 location（实测 126 条全为空），所以做两级回退：
# ① location 字段有值就用它；② 否则从职位名里猜城市；③ 再不行标「未标注」
_CITY_HINT = re.compile(r'(北京|上海|深圳|广州|杭州|成都|南京|武汉|西安|天津|苏州|重庆)')


def _district(row):
    loc = str(row.get('location', '') or '')
    if '·' in loc:
        parts = loc.split('·')
        if len(parts) > 1 and parts[1].strip():
            return parts[1].strip()
    if loc.strip():
        return loc.strip()
    m = _CITY_HINT.search(str(row.get('title', '')))
    if m:
        return m.group(1) + '（从职位名推测）'
    return '未标注'


df_sorted['district'] = df_sorted.apply(_district, axis=1)

# ---------- 过滤明确面向更低届别的岗位 ----------
# 你是 2026 届，标题里写「27届/28届/2027校招」的多是给在读生的，投了浪费。
#
# 注意：正则必须要求「届」或「20XX+校招」这种完整写法。
# 早期写成 r'2[7-9]\s*届|202[7-9]' 会误伤职位编号（如 (J10581)、(J10125)），
# 一口气误杀了 42 条，所以这里把模式收紧。
_NEXT_YEAR = re.compile(
    r'2[7-9]\s*届|'                                  # 27届 / 28届 / 29届
    r'2[7-9]\s*(校招|校园招聘|秋招|春招)|'              # 27校招 / 27秋招（没有「届」字的写法）
    r'20(2[7-9])\s*年?\s*(届|校招|校园招聘|秋招|春招|培训生|实习生)|'   # 2027届 / 2027校招
    r'20(2[7-9])\s*[-_·]?\s*(北京|上海|深圳|广州|杭州|成都|南京|武汉|西安|天津|苏州|重庆|东莞|合肥|长沙|郑州|青岛|厦门|福州|济南|大连|沈阳|昆明|南昌|贵阳|南宁|海口|石家庄|太原|兰州|银川|西宁|乌鲁木齐|呼和浩特|哈尔滨|长春)|'  # 2027北京 / 2027-北京
    r'20(2[7-9])[^0-9]{0,4}(培训生|管培生|实习生|校招|校园招聘|秋招|春招|届)'   # 2027科技培训生 / 2027届
)
_hit = df_sorted['title'].astype(str).str.contains(_NEXT_YEAR, regex=True, na=False)
if _hit.any():
    _dropped = df_sorted.loc[_hit, 'title'].astype(str).tolist()
    print(f"\n🔍 剔除面向更低届别的岗 {len(_dropped)} 条（你是 2026 届）：")
    for _t in _dropped[:8]:
        print(f"    ✂ {_t[:46]}")
    if len(_dropped) > 8:
        print(f"    … 另有 {len(_dropped) - 8} 条")
    df_sorted = df_sorted[~_hit]

# ---------- 招呼语生成 ----------
def generate_greeting_with_ai(row):
    if not DEEPSEEK_CONFIG['api_key'] or DEEPSEEK_CONFIG['api_key'] == 'sk-你的DeepSeek API密钥':
        return _fallback_greeting(row)

    title = row.get('title', '该岗位')
    jd_text = row.get('jd', '') or row.get('title', '') + ' ' + row.get('tags', '')
    if len(jd_text) > 800:
        jd_text = jd_text[:800] + '...'

    tech_str = '、'.join(MY_PROFILE.get('tech_skills', [])[:6])
    my_summary = MY_PROFILE.get('personal_summary', '').strip()[:320]
    style = MY_PROFILE.get('greeting_style', 'professional')
    style_map = {
        'professional': '正式、专业、诚恳',
        'casual': '轻松、友好、有人情味',
        'enthusiastic': '热情、积极主动、有冲劲'
    }
    style_desc = style_map.get(style, '正式、专业')

    opening = _pick_opening(row)

    prompt = f"""请根据以下信息，为求职者撰写一段发给HR的打招呼语（80-95字，不要超过100字）：

【求职者背景】
{my_summary}
掌握技术：{tech_str}
学历：{MY_PROFILE['education']}

【目标岗位】
职位：{title}
岗位描述：{jd_text}

【要求】
1. **开头必须用下面指定的这一句，原样照抄，不要改写：**
   「{opening}」
   **不要出现"2026届""应届生""本科毕业生"等身份标签**，除非 JD 里明确写了"欢迎应届生""校招""应届"。
2. 第二句讲一个具体项目成果，不要说"我熟悉Python"这种空话，也不要罗列技能。
3. 第三句表达"能快速上手"这个岗位的工作，用"期待有机会进一步沟通"结尾。
4. 不要出现"GitHub""开源""代码可查"等词，也不要写任何网址。
5. 不要写"面试回复率""回复率提升"这类个人求职数据。
6. 语气：{style_desc}，不要浮夸。
7. 直接输出招呼语正文，不要加任何额外说明或引号。

招呼语："""

    try:
        headers = {
            'Authorization': f'Bearer {DEEPSEEK_CONFIG["api_key"]}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': DEEPSEEK_CONFIG['model'],
            'messages': [
                {'role': 'system', 'content': '你是一位专业的求职顾问，擅长帮求职者写出真诚、有吸引力的打招呼语。'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.8,
            'max_tokens': 250
        }

        print(f"⏳ 请求: {row.get('title', '')[:20]}...", end='', flush=True)
        response = requests.post(
            DEEPSEEK_CONFIG['base_url'],
            headers=headers,
            json=payload,
            timeout=20,
            # DeepSeek 是国内服务，必须绕过系统代理。
            # 走代理会被绕到境外出口，导致 SSL 握手中断（实测 SSLError）。
            proxies={'http': None, 'https': None},
        )
        print(" 完成")
        if response.status_code == 200:
            result = response.json()
            greeting = result['choices'][0]['message']['content'].strip()
            return _cap_length(greeting.strip('"').strip())
        print(f" 状态码 {response.status_code}，使用模板")
        return _fallback_greeting(row, opening)
    except requests.exceptions.Timeout:
        print(" 超时，使用模板")
        return _fallback_greeting(row, opening)
    except Exception as e:
        print(f" 异常: {type(e).__name__}，使用模板")
        return _fallback_greeting(row, opening)


# ---------- 招呼语开头轮换 ----------
# 实测把「开头要多样化」写进 prompt 时模型会 100% 固定用同一种，
# 所以由 Python 轮换选定，prompt 里只要求「原样照抄」。
_OPENINGS = [
    '您好，我是马丁。',
    '您好，看到贵司在招{title}。',
    '您好，我是马丁，看到贵司的{title}岗位。',
]
_OPENING_FRESH = '我是马丁，2026届电子信息工程专业，想投递{title}。'
_FRESH_PATTERN = re.compile(r'应届|校招|校园招聘|欢迎应届')
_opening_seq = []
_opening_i = [0]

MAX_GREETING_LEN = 110


def _pick_opening(row):
    title = str(row.get('title', '该岗位') or '该岗位').strip()
    opts = list(_OPENINGS)
    if _FRESH_PATTERN.search(str(row.get('jd', ''))):
        opts.append(_OPENING_FRESH)
    if not _opening_seq:
        import random
        n = len(opts)
        _opening_seq.extend(opts[i % n] for i in range(n * 4))
        random.shuffle(_opening_seq)
    return _opening_seq.pop(0).replace('{title}', title)


def _cap_length(text, limit=MAX_GREETING_LEN):
    """模型无视字数限制（实测仍会写到 115 字），代码兜底：
    超长时回退到最后一个句末，避免硬切半句话。"""
    text = str(text or '').strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    for ch in ['。', '！', '？', '；']:
        idx = head.rfind(ch)
        if idx >= limit * 0.5:
            return head[:idx + 1]
    return head.rstrip('，,、 ') + '。'


def _fallback_greeting(row, opening=None):
    """API 不可用时的降级模板。开头沿用 Python 选定的那句，避免整批一模一样。"""
    name = MY_PROFILE['your_name']
    opening = opening or f"您好，我是{name}。"
    body = ("我独立做过完整的项目落地：从方案设计、编码实现到异常分支处理都自己走过一遍，"
            "能把事情推到底。这个岗位我很感兴趣，期待有机会进一步沟通。")
    return _cap_length(opening + body)

print("\n🤖 正在生成招呼语（如未配置API则使用模板）...")
_greetings = []
for _i, (_idx, _row) in enumerate(df_sorted.iterrows(), 1):
    print(f"  [{_i:3d}] {str(_row.get('title', ''))[:26]}", end='')
    _greetings.append(generate_greeting_with_ai(_row))
    print()
df_sorted = df_sorted.copy()
df_sorted['greeting'] = _greetings

# ---------- 输出结果 ----------
print("\n" + "=" * 80)
print("📋 猎聘投递清单 TOP 10（含招呼语预览）")
print("=" * 80 + "\n")

for i, (idx, row) in enumerate(df_sorted.head(10).iterrows(), 1):
    act = row.get('active_tag') or '—'
    comp = f"  ⚠{row['company_tag']}" if row.get('company_tag') else ''
    print(f"{i:2d}. {row['match_score']:5.1f} [{act:8s}] {str(row.get('boss_name',''))[:22]:22s} | {row['title'][:24]}{comp}")
    print(f"    薪资: {row.get('salary','')}  地区: {row.get('district','') or '未标注'}")
    print(f"    {row.get('detail_scores','')}")
    print(f"    💬 {str(row.get('greeting',''))[:80]}")
    print("-" * 74)

# ---------- 导出 ----------
output_cols = ['boss_name', 'active_tag', 'company_tag', 'title', 'salary', 'salary_k',
               'district', 'company_scale', 'company_industry', 'job_link',
               'match_score', 'detail_scores', 'greeting']
output_cols = [c for c in output_cols if c in df_sorted.columns]
df_sorted[output_cols].to_csv('apply_ready_list_liepin.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ CSV 已保存: apply_ready_list_liepin.csv（{len(df_sorted)} 条）")

# ---------- HTML面板 ----------
html_content = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>猎聘 投递助手</title>
<style>
    body { font-family: 'Segoe UI', sans-serif; background: #f5f7fa; padding: 20px; }
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e4e8; }
    th { background: #2c3e50; color: white; font-weight: 600; }
    tr:hover { background: #f1f5f9; }
    .match-score { font-weight: bold; color: #2c3e50; }
    .btn-copy { background: #3498db; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; }
    .btn-copy:hover { background: #2980b9; }
    .btn-link { background: #27ae60; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 14px; }
    .btn-link:hover { background: #1e8449; }
    .toast { position: fixed; bottom: 20px; right: 20px; background: #2ecc71; color: white; padding: 10px 20px; border-radius: 6px; display: none; }
    .greeting-preview { font-size: 13px; color: #555; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: inline-block; }
</style>
</head>
<body>
<h2>📋 猎聘 AI 匹配投递清单</h2>
<p>按匹配分从高到低排序，点击"复制招呼语"按钮，再点击岗位链接，到猎聘粘贴发送。</p>
<table id="jobTable">
<thead>
<tr>
    <th>匹配分</th>
    <th>HR状态</th>
    <th>公司</th>
    <th>职位</th>
    <th>薪资</th>
    <th>地区</th>
    <th>操作</th>
</tr>
</thead>
<tbody>
"""

for _, row in df_sorted.head(50).iterrows():
    boss_name = str(row.get('boss_name', '未知公司'))
    title = str(row.get('title', '未知职位'))
    salary = str(row.get('salary', '面议'))
    district = str(row.get('district', '') or '')
    score = row['match_score']
    link = str(row.get('job_link', '#') or '#')
    greeting = str(row['greeting']).replace('"', '&quot;').replace('\n', ' ')
    preview = greeting[:30] + '...' if len(greeting) > 30 else greeting

    # HR 活跃状态 + 公司可信度标签
    act_tag = str(row.get('active_tag', '') or '')
    comp_tag = str(row.get('company_tag', '') or '')
    if '在线' in act_tag or '刚刚' in act_tag:
        act_html = f'<span style="color:#28a745;font-weight:600">● {act_tag}</span>'
    elif act_tag.endswith('天前'):
        act_html = f'<span style="color:#dc3545">{act_tag}</span>'
    elif act_tag.endswith('小时前'):
        act_html = f'<span style="color:#e67e22">{act_tag}</span>'
    else:
        act_html = '<span style="color:#bbb">未知</span>'
    if comp_tag:
        act_html += f'<br><span style="color:#dc3545;font-size:11px">⚠ {comp_tag}</span>'

    html_content += f"""
    <tr>
        <td class="match-score">{score:.1f}</td>
        <td>{act_html}</td>
        <td>{boss_name}</td>
        <td>{title}</td>
        <td>{salary}</td>
        <td>{district}</td>
        <td>
            <button class="btn-copy" data-greeting="{greeting}" onclick="copyGreeting(this)">📋 复制招呼语</button>
            <a href="{link}" target="_blank" class="btn-link">🔗 打开链接</a>
            <span class="greeting-preview" title="{greeting}">{preview}</span>
        </td>
    </tr>
    """

html_content += """
</tbody></table>
<div id="toast" class="toast">✅ 招呼语已复制，快去粘贴吧！</div>
<script>
function copyGreeting(btn) {
    const greeting = btn.getAttribute('data-greeting');
    navigator.clipboard.writeText(greeting).then(() => {
        const toast = document.getElementById('toast');
        toast.style.display = 'block';
        setTimeout(() => { toast.style.display = 'none'; }, 2000);
    }).catch(() => {
        const textarea = document.createElement('textarea');
        textarea.value = greeting;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        const toast = document.getElementById('toast');
        toast.style.display = 'block';
        setTimeout(() => { toast.style.display = 'none'; }, 2000);
    });
}
</script>
</body>
</html>
"""

with open('apply_helper_liepin.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("\n🌐 已生成猎聘辅助投递面板：apply_helper_liepin.html")
print("双击打开该文件，即可一键复制招呼语并打开链接。")

print("\n" + "=" * 60)
print("✅ 猎聘投递清单已保存至: apply_ready_list_liepin.csv")
print(f"📌 共生成 {len(df_sorted)} 条招呼语")
print("\n💡 下一步：")
print("   1. 打开 apply_helper_liepin.html（双击用浏览器打开）")
print("   2. 点击'复制招呼语' → 点击'打开链接' → 在猎聘粘贴发送")
print("   3. 按匹配分从高到低依次投递，每天 10-15 家效果最佳。")